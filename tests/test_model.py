from __future__ import annotations

import numpy as np
import pytest
import torch
from arena_planners.bridge.discrete import COMMANDS

import planner
from model_navila import (
    HIDDEN_SIZE,
    IMAGE_TOKEN_INDEX,
    MM_HIDDEN_SIZE,
    DownSampleBlock,
    MultimodalProjector,
    build_prompt,
    resize_frame,
    tokenizer_image_token,
)


def test_projector_merges_729_patches_into_196_tokens():
    torch.manual_seed(0)
    projector = MultimodalProjector()
    out = projector(torch.randn(2, 729, MM_HIDDEN_SIZE))
    assert out.shape == (2, 196, HIDDEN_SIZE)
    assert torch.isfinite(out).all()


def test_projector_parameters_match_the_checkpoint_layout():
    shapes = {k: tuple(v.shape) for k, v in MultimodalProjector().state_dict().items()}
    assert shapes == {
        "layers.1.weight": (4608,),
        "layers.1.bias": (4608,),
        "layers.2.weight": (4096, 4608),
        "layers.2.bias": (4096,),
        "layers.4.weight": (4096, 4096),
        "layers.4.bias": (4096,),
    }


def test_downsample_pads_odd_grids_and_merges_two_by_two_blocks_column_major():
    grid = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1) + 1
    out = DownSampleBlock()(grid)
    assert out.shape == (1, 4, 4)
    assert out[0].tolist() == [[1, 2, 4, 5], [7, 8, 0, 0], [3, 0, 6, 0], [9, 0, 0, 0]]


def test_image_features_splice_in_at_the_markers(tiny_model):
    ids = torch.tensor([128000, 128000, 10, IMAGE_TOKEN_INDEX, 198, IMAGE_TOKEN_INDEX, 20])
    features = torch.randn(2, 3, 32)
    embeds = tiny_model.embed(ids, features)
    table = tiny_model.llm.get_input_embeddings().weight
    assert embeds.shape == (1, 5 + 2 * 3, 32)
    torch.testing.assert_close(embeds[0, :3], table[[128000, 128000, 10]])
    torch.testing.assert_close(embeds[0, 3:6], features[0])
    torch.testing.assert_close(embeds[0, 6], table[198])
    torch.testing.assert_close(embeds[0, 7:10], features[1])
    torch.testing.assert_close(embeds[0, 10], table[20])


def test_marker_count_must_match_the_frames(tiny_model):
    ids = torch.tensor([128000, IMAGE_TOKEN_INDEX, 5])
    with pytest.raises(ValueError, match="1 image tokens for 2 images"):
        tiny_model.embed(ids, torch.randn(2, 3, 32))


def test_eight_frames_fill_1568_image_positions(tiny_model, frame: np.ndarray):
    frames = [resize_frame(frame)] * 8
    ids = tokenizer_image_token(build_prompt("go to the door"), tiny_model.tokenizer, IMAGE_TOKEN_INDEX, "pt")
    features = tiny_model.encode_images(tiny_model.preprocess(frames))
    assert features.shape == (8, 196, 32)
    embeds = tiny_model.embed(ids, features)
    assert embeds.shape[1] == ids.shape[0] - 8 + 8 * 196


def test_random_model_answers_are_text_and_steps_stay_valid(tiny_model, frame: np.ndarray):
    agent = planner.Navila(tiny_model)
    agent.reset({"instruction": "walk past the sofa and stop at the door", "signal": "arrived"})
    for _ in range(3):
        step = agent.step({"rgb": frame, "robot_pose": np.zeros(3)})
        assert step.command in COMMANDS or step.signal == "arrived"
    assert len(agent.frames) == 3
    assert agent.answers
    assert all(isinstance(a, str) for a in agent.answers)


def test_generation_is_deterministic(tiny_model, frame: np.ndarray):
    frames = [resize_frame(frame)] * 8
    assert tiny_model.generate(frames, "go to the door") == tiny_model.generate(frames, "go to the door")
