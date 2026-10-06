from __future__ import annotations

import numpy as np
import pytest
from arena_planners.bridge.discrete import COMMANDS, check_amount

import planner


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The next action is move forward 75 cm.", ("forward", 75)),
        ("The next action is move forward 25 cm.", ("forward", 25)),
        ("The next action is move forward 100 cm.", ("forward", 100)),
        ("The next action is move forward 40 cm.", ("forward", 50)),
        ("The next action is move forward 90 cm.", ("forward", 75)),
        ("The next action is move forward.", ("forward", 25)),
        ("THE NEXT ACTION IS MOVE FORWARD 50 CM.", ("forward", 25)),
        ("The next action is turn left 30 degrees.", ("left", 30)),
        ("The next action is turn left 20 degrees.", ("left", 15)),
        ("The next action is turn right 90 degrees.", ("right", 90)),
        ("The next action is turn right 50 degrees.", ("right", 45)),
        ("The next action is turn right.", ("right", 15)),
        ("The next action is stop.", ("stop", 0)),
        ("Stop, the task is completed, the next action is move forward 25 cm.", ("stop", 0)),
        ("I will go to the kitchen.", None),
        ("The next action is stopping by the door.", None),
        ("", None),
    ],
)
def test_answers_parse_like_the_navila_trainer(text: str, expected: tuple[str, int] | None):
    assert planner.parse_answer(text) == expected


@pytest.mark.parametrize(
    ("action", "value", "count"),
    [("forward", 25, 1), ("forward", 75, 3), ("forward", 100, 4), ("left", 15, 1), ("right", 45, 3), ("left", 0, 1)],
)
def test_one_answer_runs_at_least_one_primitive(action: str, value: int, count: int):
    assert planner.unit_count(action, value) == count


@pytest.fixture
def navila() -> planner.Navila:
    agent = planner.Navila(None)
    agent.reset({"instruction": "walk to the door", "signal": "arrived"})
    return agent


def test_reset_without_goal_stands_still_and_keeps_no_frames(frame: np.ndarray):
    agent = planner.Navila(None)
    assert agent.step({"rgb": frame}) == planner.Step()
    assert agent.frames == []


def test_blank_instruction_stands_still(frame: np.ndarray):
    agent = planner.Navila(None)
    agent.reset({"instruction": "  "})
    assert agent.step({"rgb": frame}) == planner.Step()
    assert agent.frames == []


@pytest.mark.parametrize("rgb", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_missing_rgb_stands_still(navila: planner.Navila, rgb):
    assert navila.step({"rgb": rgb}) == planner.Step()
    assert navila.frames == []


def test_multi_unit_answer_queues_the_rest_and_records_every_frame(navila: planner.Navila, frame: np.ndarray):
    navila.observe(frame)
    assert navila.act("The next action is move forward 75 cm.") == planner.Step(command="forward")
    assert list(navila.queue) == ["forward", "forward"]
    assert navila.step({"rgb": frame}) == planner.Step(command="forward")
    assert navila.step({"rgb": frame}) == planner.Step(command="forward")
    assert not navila.queue
    assert len(navila.frames) == 3


def test_turns_queue_in_fifteen_degree_units(navila: planner.Navila, frame: np.ndarray):
    assert navila.act("The next action is turn right 45 degrees.") == planner.Step(command="right")
    assert list(navila.queue) == ["right", "right"]


def test_unparsable_answer_moves_one_unit_forward(navila: planner.Navila):
    assert navila.act("Let me look around first.") == planner.Step(command="forward")
    assert not navila.queue


@pytest.mark.parametrize(("state", "expected"), [({"signal": "arrived"}, "arrived"), ({}, "")])
def test_stop_latches_and_signals_only_when_asked(frame: np.ndarray, state: dict, expected: str):
    agent = planner.Navila(None)
    agent.reset({"instruction": "walk to the door", **state})
    agent.queue.extend(["left", "left"])
    assert agent.act("The next action is stop.") == planner.Step(signal=expected)
    assert not agent.queue
    for _ in range(3):
        assert agent.step({"rgb": frame}) == planner.Step(signal=expected)
    assert agent.frames == []


def test_reset_clears_latch_queue_and_history(navila: planner.Navila, frame: np.ndarray):
    navila.observe(frame)
    navila.act("The next action is turn left 45 degrees.")
    navila.act("The next action is stop.")
    navila.reset({"instruction": "turn around"})
    assert not navila.stopped
    assert navila.signal == ""
    assert not navila.queue
    assert navila.frames == []
    assert navila.answers == []
    assert navila.instruction == "turn around"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The next action is move forward 75 cm.", planner.Step(command="forward", amount=0.75)),
        ("The next action is move forward 100 cm.", planner.Step(command="forward", amount=1.0)),
        ("The next action is turn left 30 degrees.", planner.Step(command="left", amount=30.0)),
        ("The next action is turn right 15 degrees.", planner.Step(command="right", amount=15.0)),
    ],
)
def test_amount_execution_sends_one_sized_move(text: str, expected: planner.Step):
    agent = planner.Navila(None, execution="amount")
    agent.reset({"instruction": "walk to the door"})
    step = agent.act(text)
    assert step == expected
    assert not agent.queue
    check_amount(step.command, step.amount)


def test_every_command_is_a_bridge_command(navila: planner.Navila):
    for text in ("move forward", "turn left", "turn right"):
        assert navila.act(f"The next action is {text}.").command in COMMANDS


@pytest.mark.parametrize("execution", ["unit", ""])
def test_unknown_execution_names_the_accepted_values(execution: str):
    with pytest.raises(ValueError, match="units"):
        planner.Navila(None, execution=execution)


def test_unknown_dtype_names_the_accepted_values():
    with pytest.raises(ValueError, match="bfloat16"):
        planner.select_dtype("half")


def test_missing_weights_name_the_fetch_command(tmp_path):
    (tmp_path / "weights.yaml").write_text((planner._DIR / "weights.yaml").read_text())
    with pytest.raises(FileNotFoundError, match="arena feature planners add navila"):
        planner.load_model(tmp_path, planner.select_device("cpu"), planner.select_dtype("float32"))
