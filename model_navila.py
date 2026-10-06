# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NaVILA (https://github.com/AnjieCheng/NaVILA, commit 76b98f2) for inference on stock transformers,
# see ATTRIBUTION.md for the list of changes.
"""NaVILA inference on stock transformers: prompt, frame sampling, SigLIP encoder, projector and greedy generation."""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
    SiglipImageProcessorPil,
    SiglipVisionModel,
)

IMAGE_TOKEN_INDEX = -200
NUM_FRAMES = 8
IMAGE_SIZE = 384
SELECT_LAYER = -2
MM_HIDDEN_SIZE = 1152
HIDDEN_SIZE = 4096
TOKENIZER_MODEL_MAX_LENGTH = 4096
MAX_NEW_TOKENS = 32
BOS_TOKEN_ID = 128000
EOS_TOKEN_IDS = (128001, 128009)

SYSTEM = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)
ROLES = ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n")
SEP = "<|eot_id|>"


def build_question(instruction: str, num_frames: int = NUM_FRAMES) -> str:
    """The NaVILA VLN-CE question (evaluation/vlnce_baselines/navila_trainer.py)."""
    interleaved_images = "<image>\n" * (num_frames - 1)
    return (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        f"degree, moving forward a certain distance, or stop if the task is completed."
    )


def build_prompt(instruction: str, num_frames: int = NUM_FRAMES) -> str:
    """The llama_3 conversation of one user question and an open assistant turn (Conversation.get_prompt)."""
    messages = [(ROLES[0], build_question(instruction, num_frames)), (ROLES[1], None)]
    ret = SYSTEM + SEP
    for rid, (role, message) in enumerate(messages):
        if message:
            sep = SEP if rid < len(messages) - 1 else "<|end_of_text|>"
            ret += role + message + sep
        else:
            ret += role
    return ret


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None, lstrip=False):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if lstrip:
        offset = 1
    else:
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

    for chunk_id, x in enumerate(insert_separator(prompt_chunks, [image_token_index] * (offset + 1))):
        if chunk_id == 0 and lstrip:
            input_ids.extend(x)
        else:
            input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def black_frame() -> np.ndarray:
    """A padding frame, black at the model resolution (upstream pads 512x512 black images before resizing)."""
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def sample_and_pad_images(images, num_frames=NUM_FRAMES):
    frames = list(images)

    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, black_frame())

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    return sampled_frames


def resize_frame(rgb: np.ndarray) -> np.ndarray:
    """An RGB uint8 frame resized to 384x384 like `process_image` with `image_aspect_ratio: resize` (PIL bicubic)."""
    image = Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8, copy=False)).convert("RGB")
    return np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))


def default_image_processor() -> SiglipImageProcessorPil:
    """The checkpoint's `vision_tower/preprocessor_config.json` values."""
    return SiglipImageProcessorPil(
        do_resize=True,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        resample=Image.Resampling.BICUBIC,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


class DownSampleBlock(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        return x


class MultimodalProjector(nn.Module):
    """The `mlp_downsample` projector: 2x2 token merge, LayerNorm, Linear, GELU, Linear."""

    def __init__(self, mm_hidden_size: int = MM_HIDDEN_SIZE, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            DownSampleBlock(),
            nn.LayerNorm(mm_hidden_size * 4),
            nn.Linear(mm_hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.layers(x)


class NaVILA:
    """Vision tower, projector, Llama-3 LLM and tokenizer of one NaVILA checkpoint."""

    def __init__(
        self,
        llm: LlamaForCausalLM,
        vision_tower: SiglipVisionModel,
        projector: MultimodalProjector,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: SiglipImageProcessorPil,
    ) -> None:
        self.llm = llm.eval()
        self.vision_tower = vision_tower.eval()
        self.projector = projector.eval()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.llm.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=list(EOS_TOKEN_IDS),
            pad_token_id=tokenizer.eos_token_id,
        )

    @classmethod
    def load(cls, path: pathlib.Path, device: torch.device, dtype: torch.dtype) -> NaVILA:
        """Load the `llm/`, `vision_tower/` and `mm_projector/` folders of the checkpoint at `path`."""
        llm = LlamaForCausalLM.from_pretrained(path / "llm", dtype=dtype, attn_implementation="sdpa")
        vision_tower = SiglipVisionModel.from_pretrained(path / "vision_tower", dtype=dtype, attn_implementation="sdpa")
        projector = MultimodalProjector()
        projector.load_state_dict(load_file(path / "mm_projector" / "model.safetensors"), strict=True)
        tokenizer = AutoTokenizer.from_pretrained(path / "llm", padding_side="right")
        image_processor = SiglipImageProcessorPil.from_pretrained(path / "vision_tower")
        return cls(llm.to(device), vision_tower.to(device), projector.to(device, dtype), tokenizer, image_processor)

    @property
    def device(self) -> torch.device:
        return self.llm.device

    @property
    def dtype(self) -> torch.dtype:
        return self.llm.dtype

    def preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Pixel values `(N, 3, 384, 384)` of 384x384 RGB uint8 frames."""
        pixels = self.image_processor.preprocess([Image.fromarray(f) for f in frames], return_tensors="pt")
        return pixels["pixel_values"].to(self.device, self.dtype)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Projected image tokens `(N, 196, hidden)`: SigLIP hidden_states[-2], all patches, then the projector."""
        out = self.vision_tower(pixel_values.to(self.device, self.vision_tower.dtype), output_hidden_states=True)
        features = out.hidden_states[SELECT_LAYER].to(pixel_values.dtype)
        return self.projector(features)

    def embed(self, input_ids: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """Batch-1 input embeddings with one image's tokens spliced in at each IMAGE_TOKEN_INDEX position."""
        ids = input_ids.to(self.device)
        text_ids = ids.clone()
        text_ids[text_ids == IMAGE_TOKEN_INDEX] = 0
        text_embeds = self.llm.get_input_embeddings()(text_ids)
        positions = [-1, *torch.where(ids == IMAGE_TOKEN_INDEX)[0].tolist(), ids.shape[0]]
        if len(positions) - 2 != image_features.shape[0]:
            raise ValueError(f"{len(positions) - 2} image tokens for {image_features.shape[0]} images")
        pieces = []
        for i in range(len(positions) - 1):
            pieces.append(text_embeds[positions[i] + 1 : positions[i + 1]])
            if i < image_features.shape[0]:
                pieces.append(image_features[i].to(text_embeds.dtype))
        return torch.cat(pieces)[:TOKENIZER_MODEL_MAX_LENGTH].unsqueeze(0)

    @torch.inference_mode()
    def generate(self, frames: list[np.ndarray], instruction: str) -> str:
        """The answer text for 384x384 RGB frames (history then current) and an instruction."""
        prompt = build_prompt(instruction, len(frames))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        image_features = self.encode_images(self.preprocess(frames))
        inputs_embeds = self.embed(input_ids, image_features).to(self.dtype)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        output_ids = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        outputs = outputs.strip()
        if outputs.endswith(SEP):
            outputs = outputs[: -len(SEP)]
        return outputs.strip()
