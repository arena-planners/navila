# Attribution

## Code

`model_navila.py` is adapted from NaVILA, commit 76b98f2 (2025-08-19), https://github.com/AnjieCheng/NaVILA, Apache
License 2.0 (see LICENSE), Copyright 2024 NVIDIA CORPORATION & AFFILIATES:

- `llava/conversation.py`: the `llama_3` conversation (`llama_3_chat`) and its `SeparatorStyle.LLAMA_3` rendering in
  `Conversation.get_prompt`, as `SYSTEM`, `ROLES`, `SEP` and `build_prompt`.
- `evaluation/vlnce_baselines/navila_trainer.py`: the VLN-CE question (`build_question`) and `sample_and_pad_images`.
- `llava/mm_utils.py`: `tokenizer_image_token` verbatim, and the `image_aspect_ratio: resize` branch of
  `process_image` (`resize_frame`).
- `llava/model/multimodal_projector/base_projector.py`: `DownSampleBlock` verbatim and the `mlp_downsample` branch of
  `MultimodalProjector`.
- `llava/model/multimodal_encoder/vision_encoder.py` (`VisionTower.feature_select`, `cls_patch`) and
  `llava/model/llava_arch.py` (`encode_images`, `prepare_inputs_labels_for_multimodal`, `generate`): `NaVILA.encode_images`,
  `NaVILA.embed` and `NaVILA.generate`.

Changes against upstream:

- No `vila`/`llava` package. The LLM, vision tower and tokenizer are stock transformers 5 classes
  (`LlamaForCausalLM` and `SiglipVisionModel` with SDPA attention, `AutoTokenizer`, `SiglipImageProcessorPil`), loaded
  from the checkpoint's `llm/`, `vision_tower/` and `mm_projector/` folders. Upstream runs on transformers 4.37 with
  replaced `modeling_llama.py`.
- `prepare_inputs_labels_for_multimodal` is reduced to batch 1 without padding, labels, sequence parallelism or
  packing: the image features of frame i replace the i-th `IMAGE_TOKEN_INDEX` (-200) marker, the result is cut at
  `tokenizer_model_max_length` (4096) as upstream does.
- Generation is greedy with at most 32 new tokens and stops on `<|end_of_text|>` or `<|eot_id|>` (the checkpoint's
  `eos_token_id` list), padding with `<|eot_id|>`. Upstream adds a keyword stopping criterion on `<|eot_id|>`, which
  the EOS list already covers.
- `sample_and_pad_images` works on 384x384 uint8 arrays and pads with black 384x384 frames. Upstream pads with black
  512x512 PIL images and resizes all frames to 384x384 afterwards, which gives the same pixels.
- Decoding passes `clean_up_tokenization_spaces=False`. transformers 5 ignores the cleanup for BPE tokenizers anyway,
  upstream applied it (it only removes spaces before punctuation, which the parser does not depend on).

`planner.py` ports the action parsing and execution of `NaVILATrainer.eval` (`navila_trainer.py`): the four
case-insensitive action patterns in their order (stop first), the case-sensitive `move forward (\d+) cm` and
`turn left|right (\d+) degree` arguments with defaults 25 and 15 and snapping to 25/50/75 or 15/30/45, one VLN-CE
primitive per 25 cm or 15 degrees with at least one, the queue of the remaining primitives, and the history of every
observed frame. An answer that matches no pattern moves forward one primitive (upstream would pass `None` to the
environment).

## Weights

`a8cheng/navila-llama3-8b-8f` (revision b2294e9) on the Hugging Face Hub, the VLN-CE checkpoint of the NaVILA paper.
The model card carries no license. Its base, Efficient-Large-Model/Llama-3-VILA1.5-8B, is CC BY-NC 4.0 and derives
from Meta Llama 3 (Meta Llama 3 Community License), and the navigation training data derives from Matterport3D
(Matterport3D Terms of Use). Treat the weights as non-commercial research use only.

Cite:

```bibtex
@inproceedings{cheng2025navila,
        title={Navila: Legged robot vision-language-action model for navigation},
        author={Cheng, An-Chieh and Ji, Yandong and Yang, Zhaojing and Gongye, Zaitian and Zou, Xueyan and Kautz, Jan and B{\i}y{\i}k, Erdem and Yin, Hongxu and Liu, Sifei and Wang, Xiaolong},
        booktitle={RSS},
        year={2025}
}
```
