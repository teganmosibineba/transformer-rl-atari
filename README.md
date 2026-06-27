# Decision Transformer — Atari

Offline reinforcement learning as sequence modelling. A GPT-2 backbone conditioned on return-to-go, state, and action tokens, trained on Atari game trajectories without any environment interaction during training.

Trained on **Breakout, Pong, and Qbert** using the [JAT dataset](https://huggingface.co/datasets/jat-project/jat-dataset-tokenized) on a single Kaggle T4 GPU.

---

## Training results

These are the results from the completed 100,000 step training run:

| Metric | Value |
|--------|-------|
| Total steps | 100,000 |
| Best loss | 0.0055 |
| Final accuracy | 0.998 |
| Training time | ~6 hours (Kaggle T4 x2) |
| Games trained on | Breakout, Pong, Qbert |

Loss curve — from 1.4 (random baseline) down to 0.005:

![Training curves](training_curves.png)

Game evaluation scores (rollout eval — coming soon):

| Game | Random | DQN baseline | Decision Transformer | vs DQN |
|------|--------|-------------|---------------------|--------|
| Breakout | 1.7 | 30.5 | — | — |
| Pong | -20.7 | 18.9 | — | — |
| Qbert | 163.9 | 1152.0 | — | — |

*Eval rollouts will be added once the evaluation script is run.*

---

## How it works

Standard RL learns by interacting with an environment and updating a value function or policy gradient. Decision Transformer throws that away. It treats a trajectory as a sequence:

```
[RTG₀, s₀, a₀, RTG₁, s₁, a₁, ..., RTGₖ, sₖ, aₖ]
```

At inference time, you condition the model on a **Return-to-Go (RTG)** — the score you *want* the agent to achieve — and it predicts the action sequence that achieves it. Higher target = more ambitious play.

---

## Architecture

```
Atari frames (4×84×84)
       │
  CNN encoder    ← Conv(8,4) → Conv(4,2) → Conv(3,1) → Linear(3136→128)
       │
  State + Action + RTG embeddings  ← all projected to hidden_size=128
       │                              + timestep embedding added to each
  Interleave: [RTG₀, s₀, a₀, RTG₁, s₁, a₁, ...]
       │
  GPT-2 (6 layers, 8 heads, context K=30 timesteps = 90 tokens)
       │
  Action head  ← linear applied to state token output positions only
       │
  Action logits → CrossEntropyLoss
```

**~3M parameters.** Trained in 6 hours on a single T4.

---

## Training details

| Hyperparameter | Value |
|----------------|-------|
| Backbone | GPT-2 (6L, 8H) |
| Hidden size | 128 |
| Context length K | 30 timesteps |
| Optimizer | AdamW β=(0.9, 0.95) |
| Learning rate | 6e-4 with linear warmup + cosine decay |
| Warmup steps | 2,500 |
| Total steps | 100,000 |
| Batch size | 64 |
| Gradient clip | 0.25 |
| RTG discount γ | 1.0 (undiscounted) |
| Data | JAT dataset (jat-project/jat-dataset-tokenized) |
| Hardware | Kaggle T4 x2 |

---

## Quickstart

**Load the checkpoint:**

```python
from huggingface_hub import hf_hub_download
import torch

ckpt_path = hf_hub_download(
    repo_id="TeganJegede/transformer-rl-atari",
    filename="checkpoints/dt_step075000_loss0.0055.pt",
    repo_type="model",
)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
```

**Run the training notebook:**

Upload `transformer_rl_atari.ipynb` to Kaggle, set accelerator to GPU T4 x2, Run All.

---

## Repository structure

```
transformer-rl-atari/
├── transformer_rl_atari.ipynb   # full training notebook (Kaggle T4)
├── README.md
└── checkpoints/
    └── dt_step075000_loss0.0055.pt   # best checkpoint
```

---

## Key design decisions

**Why undiscounted RTG (γ=1.0)?** The DT sees the full future sum directly — discounting would shrink target RTGs and hurt conditioning signal.

**Why predict from state token output?** The state token has attended to both RTG and observation before the action token is produced — it's the right context for predicting the next action.

**Why offline RL?** No environment interaction during training means no reward hacking, no exploration instability. The model learns purely from recorded trajectories. This has direct safety implications — a model that never explores cannot discover unintended high-reward strategies through trial and error.

---

## Connection to AI safety

Training stability is a safety-relevant property. This project is a natural extension of my HRA-RL research on reward stability in multimodal agents. Offline RL sidesteps a class of reward hacking risks by removing live environment interaction entirely — consistent with the empirical framing in my [ICML 2026 AgenticUQ Workshop paper](https://huggingface.co/TeganJegede).

---

## Citation

If this work is useful:

```bibtex
@misc{jegede2026dt,
  author = {Jegede, Tegan},
  title  = {Decision Transformer — Atari},
  year   = {2026},
  url    = {https://github.com/teganmosibineba/transformer-rl-atari}
}
```

Original paper: Chen et al., [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345), NeurIPS 2021.

---

## Author

**Tegan Jegede** — MSc Computer Science, Nile University  
Research: hybrid reward architectures, multimodal agent factuality, AI safety  
[HuggingFace](https://huggingface.co/TeganJegede) · [GitHub](https://github.com/teganmosibineba)
