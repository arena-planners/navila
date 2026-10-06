"""NaVILA (navila-llama3-8b-8f) wrapper for the arena_planners bridge."""

from __future__ import annotations

import collections
import logging
import pathlib
import re
import time

import numpy as np
import torch
from arena_planners import weights
from arena_planners.sdk import Step, load_manifest, main_loop

from model_navila import NUM_FRAMES, NaVILA, resize_frame, sample_and_pad_images

_DIR = pathlib.Path(__file__).parent
_FETCH_HINT = "run `arena feature planners add navila` (or `python -m arena_planners fetch navila`)"

FORWARD_CM = 25
TURN_DEG = 15
_FORWARD_CHOICES = (25, 50, 75)
_TURN_CHOICES = (15, 30, 45)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("stop", re.compile(r"\bstop\b", re.IGNORECASE)),
    ("forward", re.compile(r"\bis move forward\b", re.IGNORECASE)),
    ("left", re.compile(r"\bis turn left\b", re.IGNORECASE)),
    ("right", re.compile(r"\bis turn right\b", re.IGNORECASE)),
)
_ARGUMENTS: dict[str, tuple[re.Pattern[str], int, tuple[int, ...]]] = {
    "forward": (re.compile(r"move forward (\d+) cm"), FORWARD_CM, _FORWARD_CHOICES),
    "left": (re.compile(r"turn left (\d+) degree"), TURN_DEG, _TURN_CHOICES),
    "right": (re.compile(r"turn right (\d+) degree"), TURN_DEG, _TURN_CHOICES),
}

EXECUTIONS = ("units", "amount")
DTYPES: dict[str, torch.dtype] = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

_log = logging.getLogger("navila")


def parse_answer(text: str) -> tuple[str, int] | None:
    """Action and size of an answer: `("stop", 0)`, or a command with centimeters or degrees, None if unmatched."""
    action = next((name for name, pattern in _PATTERNS if pattern.search(text)), None)
    if action is None:
        return None
    if action == "stop":
        return "stop", 0
    pattern, default, choices = _ARGUMENTS[action]
    match = pattern.search(text)
    value = int(match.group(1)) if match else default
    if value % default:
        value = min(choices, key=lambda x: abs(x - value))
    return action, value


def unit_count(action: str, value: int) -> int:
    """Number of 25 cm or 15 degree primitives VLN-CE runs for one answer, at least one."""
    return max(1, value // (FORWARD_CM if action == "forward" else TURN_DEG))


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def select_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"navila: dtype {name!r} not in {sorted(DTYPES)}")
    return DTYPES[name]


def load_model(planner_dir: pathlib.Path, device: torch.device, dtype: torch.dtype) -> NaVILA:
    if absent := weights.missing(planner_dir):
        raise FileNotFoundError(f"navila weights missing at {planner_dir} ({', '.join(absent)}), {_FETCH_HINT}")
    return NaVILA.load(planner_dir / "model", device, dtype)


class Navila:
    def __init__(self, model: NaVILA | None, execution: str = "units") -> None:
        if execution not in EXECUTIONS:
            raise ValueError(f"navila: execution {execution!r} not in {list(EXECUTIONS)}")
        self.model = model
        self.execution = execution
        self._warned: set[str] = set()
        self.reset(None)

    def reset(self, initial_state: dict | None) -> None:
        self.instruction = ""
        self.signal = ""
        self.stopped = False
        self.frames: list[np.ndarray] = []
        self.queue: collections.deque[str] = collections.deque()
        self.answers: list[str] = []
        if initial_state is None:
            return
        self.signal = str(initial_state.get("signal", ""))
        self.instruction = str(initial_state.get("instruction", "")).strip()
        if not self.instruction:
            self._warn_once("empty", "navila: no instruction for this episode, standing still")

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _log.warning(msg)

    def _stopped_step(self) -> Step:
        return Step(signal=self.signal) if self.signal else Step()

    def observe(self, rgb: np.ndarray) -> None:
        """Append one camera frame to the episode history at the model resolution."""
        self.frames.append(resize_frame(np.asarray(rgb)))

    def model_frames(self) -> list[np.ndarray]:
        """The eight frames of the next query: seven sampled from the history, then the current one."""
        return sample_and_pad_images(self.frames, NUM_FRAMES)

    def act(self, answer: str) -> Step:
        """Turn a model answer into the next Step, queueing the remaining primitives in `units` execution."""
        self.answers.append(answer)
        parsed = parse_answer(answer)
        if parsed is None:
            self._warn_once("unmatched", f"navila: unparsable answer {answer!r}, moving forward one step")
            parsed = ("forward", FORWARD_CM)
        action, value = parsed
        if action == "stop":
            self.stopped = True
            self.queue.clear()
            return self._stopped_step()
        count = unit_count(action, value)
        if self.execution == "amount":
            amount = count * (FORWARD_CM / 100.0 if action == "forward" else float(TURN_DEG))
            return Step(command=action, amount=amount)
        self.queue.extend([action] * (count - 1))
        return Step(command=action)

    def query(self) -> str:
        assert self.model is not None
        start = time.perf_counter()
        answer = self.model.generate(self.model_frames(), self.instruction)
        _log.info("navila: %r after %d frames in %.2f s", answer, len(self.frames), time.perf_counter() - start)
        return answer

    def step(self, features: dict) -> Step:
        if self.stopped:
            return self._stopped_step()
        if not self.instruction:
            return Step()
        rgb = features.get("rgb")
        if rgb is None or np.asarray(rgb).size == 0:
            self._warn_once("sensors", "navila: rgb missing, standing still")
            return Step()
        self.observe(rgb)
        if self.queue:
            return Step(command=self.queue.popleft())
        return self.act(self.query())


_planner: Navila | None = None


def on_init(config: dict) -> None:
    global _planner
    device = select_device(str(config.get("device", "auto")))
    dtype = select_dtype(str(config.get("dtype", "float16")))
    execution = str(config.get("execution", "units"))
    _planner = Navila(load_model(_DIR, device, dtype), execution)
    _log.warning("navila: loaded on %s as %s, %s execution", device, dtype, execution)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _planner is not None
    _planner.reset(initial_state)


def step(features: dict) -> Step:
    assert _planner is not None
    return _planner.step(features)


if __name__ == "__main__":
    logging.basicConfig()
    manifest = load_manifest(_DIR / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset, on_init=on_init)
