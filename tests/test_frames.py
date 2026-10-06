from __future__ import annotations

import numpy as np
import pytest
import torch

from model_navila import IMAGE_SIZE, black_frame, resize_frame, sample_and_pad_images


def _tagged(n: int) -> list[np.ndarray]:
    return [np.full((IMAGE_SIZE, IMAGE_SIZE, 3), i + 1, dtype=np.uint8) for i in range(n)]


def _tags(frames: list[np.ndarray]) -> list[int]:
    return [int(f[0, 0, 0]) - 1 for f in frames]


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, [-1, -1, -1, -1, -1, -1, -1, 0]),
        (3, [-1, -1, -1, -1, -1, 0, 1, 2]),
        (7, [-1, 0, 1, 2, 3, 4, 5, 6]),
        (8, [0, 1, 2, 3, 4, 5, 6, 7]),
        (9, [0, 1, 2, 3, 4, 5, 6, 8]),
        (20, [0, 2, 5, 8, 10, 13, 16, 19]),
        (100, [0, 14, 28, 42, 56, 70, 84, 99]),
    ],
)
def test_history_samples_seven_frames_over_all_frames_then_the_current_one(count: int, expected: list[int]):
    frames = sample_and_pad_images(_tagged(count))
    assert len(frames) == 8
    assert _tags(frames) == expected


def test_padding_frames_are_black_at_model_resolution():
    pad = sample_and_pad_images(_tagged(1))[0]
    assert pad.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert pad.dtype == np.uint8
    assert not pad.any()


def test_sampling_leaves_the_history_untouched():
    frames = _tagged(2)
    sample_and_pad_images(frames)
    assert len(frames) == 2


def test_frames_resize_to_384_rgb(frame: np.ndarray):
    out = resize_frame(frame)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert out.dtype == np.uint8


def test_rgba_frames_drop_alpha(frame: np.ndarray):
    rgba = np.concatenate([frame, np.full((512, 512, 1), 7, dtype=np.uint8)], axis=-1)
    np.testing.assert_array_equal(resize_frame(rgba), resize_frame(frame))


def test_pixels_scale_to_minus_one_one(tiny_model, frame: np.ndarray):
    frames = [resize_frame(frame), black_frame()]
    pixels = tiny_model.preprocess(frames)
    assert pixels.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    expected = torch.from_numpy(frames[0]).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
    torch.testing.assert_close(pixels[0], expected, atol=1e-5, rtol=0)
    assert torch.all(pixels[1] == -1.0)
