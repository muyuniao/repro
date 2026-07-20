# GECO for Hateful Meme Classification

Official code release for the CVPR 2026 paper:

**Tackling Model Bias via Game-theoretic Multi-agent Collaboration Framework for Hateful Meme Classification**

This repository implements a five-player game-theoretic collaboration framework for hateful meme classification. The model combines:

- LLaVA hidden states
- Qwen hidden states
- CLIP image-token sequence features
- CLIP text-token sequence features
- Gemma hidden states

The final classifier is trained with a Nash-advantage linear objective and an optional Jeffreys-beta regularizer on the fused player.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── run_geco.py
└── train.py
```

## Method Overview

The framework contains five decision-making players:

- `pi_L`: LLaVA branch
- `pi_Q`: Qwen branch
- `pi_C`: CLIP multimodal sequence branch
- `pi_G`: Gemma branch
- `pi_F`: fused final classifier

Each branch first projects its backbone feature into a shared latent space. The CLIP branch uses a transformer encoder to jointly model image and text token sequences, then fuses the visual and textual class tokens with a learned gate. The training objective models inter-player cooperation through:

- single-player correctness rewards
- pairwise cooperation rewards
- an all-correct cooperation bonus
- optional entropy-style game regularization
- optional Jeffreys-beta divergence regularization for `pi_F`

## Data Format

The training script expects a JSON array. Each sample should look like:

```json
{
  "id": "sample_0001",
  "hidden_state_file": "llava/sample_0001.pt",
  "hidden_state_file2": "qwen/sample_0001.pt",
  "hidden_state_file3": "clip_image/sample_0001.pt",
  "hidden_state_file4": "clip_text/sample_0001.pt",
  "hidden_state_file5": "gemma/sample_0001.pt",
  "label": 1
}
```

Notes:

- `hidden_state_file` / `hidden_state_file2` / `hidden_state_file5` can be either a single token vector or a token sequence. The script uses the last token.
- `hidden_state_file3` and `hidden_state_file4` must be CLIP sequence tensors with feature dimension `768`.
- The script uses `json.load`, so the annotation file must be a valid JSON array instead of JSONL.

## Configuration

All hyperparameters are defined in [`run_geco.py`](run_geco.py).

Important fields:

- `train_json`: path to the training annotations
- `test_json`: path to the test annotations
- `data_dir`: root directory for all hidden-state `.pt` files
- `ckpt`: checkpoint output path
- `llava_dim`, `qwen_dim`, `gemma_dim`: input dimensions for different backbones
- `xhat_mode`, `tau_reg`, `pair_bonus`, `coop_bonus`: game objective controls
- `kl_on`, `kl_ref`, `kl_beta`, `kl_target`, `kl_mix`: Jeffreys-beta regularization controls

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Edit the paths and hyperparameters in `cfg` inside `run_geco.py`, then run:

```bash
python run_geco.py
```

The training logic stays in [`train.py`](train.py). You only need to modify the configuration in [`run_geco.py`](run_geco.py).

## Metrics

The script reports:

- Accuracy
- Macro F1
- ROC-AUC

