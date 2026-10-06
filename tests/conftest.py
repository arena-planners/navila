from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, SiglipVisionConfig, SiglipVisionModel

import planner
from model_navila import BOS_TOKEN_ID, EOS_TOKEN_IDS, MultimodalProjector, NaVILA, default_image_processor

TOKENIZER_DIR = planner._DIR / "model" / "llm"


@pytest.fixture(scope="session")
def tokenizer():
    if not (TOKENIZER_DIR / "tokenizer.json").is_file():
        pytest.skip(f"tokenizer missing at {TOKENIZER_DIR}, run `arena feature planners add navila`")
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, padding_side="right")


@pytest.fixture(scope="session")
def tiny_model(tokenizer) -> NaVILA:
    torch.manual_seed(0)
    llm = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128259,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=8192,
            rope_theta=500000.0,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=EOS_TOKEN_IDS[1],
        )
    )
    vision = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=384,
            patch_size=14,
        )
    )
    return NaVILA(llm, vision, MultimodalProjector(16, 32), tokenizer, default_image_processor())


@pytest.fixture(scope="session")
def frame() -> np.ndarray:
    rows = np.linspace(0.0, 1.0, 512, dtype=np.float32)[:, None, None]
    cols = np.linspace(0.0, 1.0, 512, dtype=np.float32)[None, :, None]
    tint = np.array([0.9, 0.8, 0.6], dtype=np.float32)
    noise = np.random.default_rng(0).normal(0.0, 0.03, (512, 512, 3)).astype(np.float32)
    image = np.where(rows < 0.45, 0.85 * tint, 0.35 + 0.3 * cols * tint) + noise
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
