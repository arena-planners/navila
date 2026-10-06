from __future__ import annotations

import os
import pathlib
import time

import numpy as np
import pytest
import torch
from arena_planners import weights
from arena_planners.bridge.discrete import COMMANDS, check_amount
from PIL import Image

import planner

_GIF = pathlib.Path(os.environ.get("NAVILA_EPISODE_GIF", "/nonexistent"))
_PANE = (5, 5, 490, 370)
_LAST_FRAME = 99
_STEPS = 24
_MIN_FREE_GB = 30.0
_INSTRUCTION = (
    "You are in a bedroom. Turn around to the left until you see a door leading out into a hallway, go through it. "
    "Hang a right and walk between the island and the couch on your left. When you are between the second and third "
    "chairs for the island stop."
)


def _episode_frames() -> list[np.ndarray]:
    """512x512 center crops of the egocentric pane of NaVid-VLN-CE's `rxr_teaser.gif` before its success banner."""
    gif = Image.open(_GIF)
    frames = []
    for index in np.linspace(0, _LAST_FRAME, _STEPS).round().astype(int):
        gif.seek(int(index))
        pane = gif.convert("RGB").crop(_PANE)
        side = min(pane.size)
        left, top = (pane.width - side) // 2, (pane.height - side) // 2
        frames.append(np.asarray(pane.crop((left, top, left + side, top + side)).resize((512, 512))))
    return frames


@pytest.fixture(scope="module")
def agent() -> planner.Navila:
    if absent := weights.missing(planner._DIR):
        pytest.skip(f"{len(absent)} weight files missing, run `arena feature planners add navila`")
    if not torch.cuda.is_available():
        pytest.skip("the 8B checkpoint needs a CUDA GPU")
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    if free_gb < _MIN_FREE_GB:
        pytest.skip(f"{free_gb:.1f} GB GPU memory free, {_MIN_FREE_GB:g} GB needed")
    dtype = planner.select_dtype(os.environ.get("NAVILA_DTYPE", "float16"))
    start = time.perf_counter()
    model = planner.load_model(planner._DIR, torch.device("cuda"), dtype)
    print(f"\nloaded {dtype} in {time.perf_counter() - start:.1f} s, {torch.cuda.memory_allocated() / 2**30:.2f} GB")
    return planner.Navila(model)


def test_real_episode_yields_well_formed_answers_and_valid_steps(agent: planner.Navila):
    if not _GIF.is_file():
        pytest.skip(f"episode gif missing at {_GIF}, set NAVILA_EPISODE_GIF to NaVid-VLN-CE's rxr_teaser.gif")
    agent.reset({"instruction": _INSTRUCTION, "signal": "arrived"})
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for k, frame in enumerate(_episode_frames()):
        queried = not agent.queue and not agent.stopped
        start = time.perf_counter()
        step = agent.step({"rgb": frame, "robot_pose": np.zeros(3)})
        elapsed = time.perf_counter() - start
        if queried:
            latencies.append(elapsed)
            print(f"step {k:2d} query {elapsed:5.2f} s {agent.answers[-1]!r} -> {step}")
        else:
            print(f"step {k:2d} {'latched' if agent.stopped else 'queued '} -> {step}")
        assert step.command in COMMANDS or step == planner.Step(signal="arrived")
        check_amount(step.command, step.amount)
    peak = torch.cuda.max_memory_allocated() / 2**30
    median, worst = np.median(latencies), max(latencies)
    print(f"{len(agent.answers)} queries, median {median:.2f} s, max {worst:.2f} s, peak {peak:.2f} GB")
    well_formed = [a for a in agent.answers if planner.parse_answer(a) is not None]
    assert len(agent.answers) >= 3
    assert len(well_formed) >= 0.75 * len(agent.answers)
    assert len(agent.frames) == _STEPS or agent.stopped
