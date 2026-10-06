# navila

NaVILA (Cheng et al., RSS 2025, "NaVILA: Legged Robot Vision-Language-Action Model for Navigation", arXiv 2412.04453)
behind the arena_planners bridge, with the VLN-CE checkpoint `a8cheng/navila-llama3-8b-8f`. A Llama-3 8B VILA model
reads the instruction and eight RGB frames (seven sampled from the episode history plus the current one) and answers in
text, such as "The next action is move forward 75 cm." or "The next action is turn left 30 degree.". The planner parses
the answer into discrete moves as NaVILA's VLN-CE evaluation does. Decoding is greedy.

```sh
arena feature planners add navila
arena launch sim:=gazebo robot:=jackal[camera=vln_navila] robot.planner:=navila robot.mobile.goal:=instruction
```

`robot.planner:=navila` selects `robot.mobile:=vla`. The planner accepts the instruction only (`goal_inputs:
[instruction]`), so `robot.mobile.goal:=instruction` is also the default. It needs a CUDA GPU with about 17 GB free in
float16 (15.8 GB of weights, 16.3 GB peak), a query takes about 0.25 s on an RTX PRO 6000 Blackwell.

## Inputs

| Datasource | Topic (jackal `camera=vln_navila`) | Model input |
|---|---|---|
| `rgb` (`image` sensor) | `vln_camera/image` | 512x512 RGB from the bridge, resized to 384x384 (PIL bicubic), scaled to [-1, 1] |
| instruction | from the goal | inserted verbatim in double quotes into NaVILA's VLN-CE prompt |

The `vln_navila` camera matches NaVILA's VLN-CE sensor: 512x512, 90 degrees HFOV, 1.25 m above the floor. Each frame
becomes 196 tokens (SigLIP so400m, second to last layer, 2x2 token merge). Episodes shorter than eight frames are padded
with black frames at the front. Longer ones sample seven history frames evenly over all frames seen so far.

## Primitives

The planner returns discrete commands (`action_type: discrete`) and the bridge executes each as a motion primitive:

| Answer | Command | Execution (`units`) |
|---|---|---|
| `... is move forward N cm` | `forward` | `N // 25` steps of 0.25 m, N snapped to 25/50/75 unless a multiple of 25 |
| `... is turn left N degree` | `left` | `N // 15` steps of 15 degrees, N snapped to 15/30/45 unless a multiple of 15 |
| `... is turn right N degree` | `right` | as `left`, clockwise |
| any answer containing `stop` | none | ends the episode for the planner |

A missing number means 25 cm or 15 degrees, and every answer runs at least one step. The planner queues the remaining
steps and returns them one per observation without querying the model, recording the camera frame of every step, which
reproduces VLN-CE's one frame per executed action. An answer that matches no pattern moves forward one step and logs a
warning once.

On `stop` the planner latches: it drops the queue, stops calling the model and, for a goto phase with `signal:
arrived`, keeps sending `arrived` until the next reset. Without an expected signal it stands still. Before an
instruction arrives (a reset without a goal) or with a blank instruction, the planner stands still without running the
model.

Instructions come from the scenario, per goto phase:

```yaml
robots:
  - start: [1.0, 1.0, 0.0]
    phases:
      - goto: [8.0, 8.0, 0.0]
        instruction: walk past the sofa and stop at the door to the kitchen
        signal: arrived
```

The goal pose decides success but never reaches the planner.

## Config

Set in `planner.yaml > config`, override per launch with `robot.mobile.config.<key>:=<value>`.

| Key | Default | Effect |
|---|---|---|
| `device` | `auto` | Torch device, `auto` picks CUDA when available. |
| `dtype` | `float16` | `float16` (NaVILA's evaluation precision), `bfloat16` (its training precision) or `float32`. |
| `execution` | `units` | `units` runs one 0.25 m or 15 degree primitive per observation, as in VLN-CE. `amount` sends each answer as one move sized by its distance or angle, so the history holds one frame per answer. |

## Tests

`pytest` in this directory runs the prompt, tokenizer, frame sampling, projector, parser and queue tests on CPU with a
randomly initialized tiny model. They need only the tokenizer files under `model/llm/` and skip without them.
`tests/test_episode.py` runs the real checkpoint over 24 frames of a Matterport walkthrough with its RxR instruction and
skips without the weights, without a CUDA GPU with 30 GB free, or without `NAVILA_EPISODE_GIF` pointing to
`VLN_CE/data/res/rxr_teaser.gif` of https://github.com/jzhzhang/NaVid-VLN-CE. `NAVILA_DTYPE` picks its precision.

## Weights and license

`arena feature planners add navila` fetches the checkpoint (about 16 GB: `llm/`, `vision_tower/`, `mm_projector/`) at
a pinned revision into `model/`, every file checked against its sha256. The model card carries no license. Its base,
Llama-3-VILA1.5-8B, is CC BY-NC 4.0 on top of Meta Llama 3, and the navigation data derives from Matterport3D: use the
weights for non-commercial research only. The code is Apache 2.0, see LICENSE and ATTRIBUTION.md. Cite Cheng et al.
2025 when using this planner:

```bibtex
@inproceedings{cheng2025navila,
        title={Navila: Legged robot vision-language-action model for navigation},
        author={Cheng, An-Chieh and Ji, Yandong and Yang, Zhaojing and Gongye, Zaitian and Zou, Xueyan and Kautz, Jan and B{\i}y{\i}k, Erdem and Yin, Hongxu and Liu, Sifei and Wang, Xiaolong},
        booktitle={RSS},
        year={2025}
}
```
