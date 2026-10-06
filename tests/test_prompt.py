from __future__ import annotations

from model_navila import IMAGE_TOKEN_INDEX, build_prompt, tokenizer_image_token

_INSTRUCTION = "Walk past the sofa and stop at the door to the kitchen."

_EXPECTED = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful language and vision assistant. You are able to understand the visual content that the user "
    "provides, and assist the user with a variety of tasks using natural language.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations "
    "<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n, and current observation <image>\n. "
    'Your assigned task is: "Walk past the sofa and stop at the door to the kitchen." '
    "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
    "degree, moving forward a certain distance, or stop if the task is completed.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def test_prompt_renders_the_llama3_conversation():
    assert build_prompt(_INSTRUCTION) == _EXPECTED


def test_prompt_keeps_the_instruction_raw():
    assert '"turn LEFT, then  stop!"' in build_prompt("turn LEFT, then  stop!")


def test_token_ids_start_with_a_double_bos(tokenizer):
    ids = tokenizer_image_token(build_prompt(_INSTRUCTION), tokenizer, IMAGE_TOKEN_INDEX)
    assert ids[:4] == [128000, 128000, 128006, 9125]
    assert ids.count(128000) == 2


def test_token_ids_hold_one_marker_per_frame_between_newlines(tokenizer):
    ids = tokenizer_image_token(build_prompt(_INSTRUCTION), tokenizer, IMAGE_TOKEN_INDEX)
    positions = [i for i, t in enumerate(ids) if t == IMAGE_TOKEN_INDEX]
    assert len(positions) == 8
    assert all(ids[p + 1] == 198 for p in positions[:7])
    assert all(b - a == 2 for a, b in zip(positions[:6], positions[1:7], strict=True))


def test_token_ids_end_with_the_open_assistant_turn(tokenizer):
    ids = tokenizer_image_token(build_prompt(_INSTRUCTION), tokenizer, IMAGE_TOKEN_INDEX)
    assert ids[-5:] == [128009, 128006, 78191, 128007, 271]


def test_text_between_markers_tokenizes_without_bos(tokenizer):
    ids = tokenizer_image_token(build_prompt(_INSTRUCTION), tokenizer, IMAGE_TOKEN_INDEX)
    text = tokenizer.decode([t for t in ids if t != IMAGE_TOKEN_INDEX])
    assert text == "<|begin_of_text|>" + _EXPECTED.replace("<image>", "")
