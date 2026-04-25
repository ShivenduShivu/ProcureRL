# ProcureRL - Procurement Negotiation RL Environment

[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/ShivenduShivu/ProcureRL)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

## Meta OpenEnv Hackathon 2026 Submission

**[Live Environment on HuggingFace Spaces](https://huggingface.co/spaces/ShivenduShivu/ProcureRL)**

**[Demo Video (< 2 min)](https://youtube.com/YOUR_VIDEO_LINK)**

**[HuggingFace Blog Post](https://huggingface.co/blog/YOUR_POST)**

**[Training Notebook (Colab)](https://colab.research.google.com/github/ShivenduShivu/ProcureRL/blob/main/notebooks/ProcureRL_Training.ipynb)**

---

## 1. The Problem

LLMs can hold a conversation, but they fail at strategic negotiation.
In procurement, a negotiator must:

- Anchor strategically below the seller's opening price
- Use competitive pressure signals to create urgency
- Trade price against delivery time and quality tier
- Never exceed the procurement policy budget ceiling

No RL training environment existed for this. We built one.

## 2. The Environment

**ProcureRL** is a multi-agent procurement negotiation environment built on
[AgenticPay](https://arxiv.org/pdf/2602.06008) and wrapped in OpenEnv-style interfaces.

| Feature | Description |
|---|---|
| Agents | Buyer (trained LLM) vs Seller (fixed scripted policy) |
| Variables | Price + delivery days + quality tier (3-dimensional) |
| Hidden info | Buyer's budget ceiling hidden from seller; seller's cost floor hidden from buyer |
| Market signal | Noisy estimate of true market price visible to buyer |
| Competitor signal | Optional competing supplier quote creates pressure |
| Constraint | Hard policy ceiling - exceeding it gives reward = -1.0 |
| Curriculum | Easy -> Medium -> Hard (price gap widens, rounds shrink) |

## 3. Results

We train Qwen2.5-3B-Instruct using GRPO (TRL + Unsloth, 4-bit QLoRA).

| Metric | Baseline | Trained | Improvement |
|---|---|---|---|
| Deal Rate | Pending Colab run | Pending Colab run | Pending |
| Mean Reward | Pending Colab run | Pending Colab run | Pending |
| Mean Savings | Pending Colab run | Pending Colab run | Pending |
| Mean Rounds | Pending Colab run | Pending Colab run | Pending |
| Constraint Violations | Pending Colab run | Pending Colab run | Pending |

The Layer 5 infrastructure is complete, but the real training run still needs to be executed in Colab on a T4 GPU. The next required step is running [notebooks/ProcureRL_Training.ipynb](C:/Users/shive/AppData/Local/Programs/META/ProcureRL/notebooks/ProcureRL_Training.ipynb) and committing the actual metrics plus final plots before submission.

### Reward Curves

![Total Reward](results/plots/reward_total.png)
*Placeholder plot path wired for the final Colab-generated artifact.*

![Reward Components](results/plots/reward_components.png)
*Placeholder plot path wired for the final Colab-generated artifact.*

![Before vs After](results/plots/before_after.png)
*Placeholder plot path wired for the final Colab-generated artifact.*

## 4. Why It Matters

Enterprise procurement represents trillions in annual spend globally.
An AI agent trained to negotiate strategically, use market data, and respect policy constraints has direct commercial value.
ProcureRL is a procurement-specific RL environment designed to make that training measurable and reproducible.

## Setup

```bash
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" pydantic
pip install -e .
```

```python
from agenticpay.openenv_adapter.procure_env_extended import ProcureEnvExtended

env = ProcureEnvExtended(difficulty="easy")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(
    "I can offer $108 for delivery within 18 days. <BUYER_PRICE>108</BUYER_PRICE>"
)
```

## API

Run the local server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Example endpoints:

- `POST /reset`
- `POST /step`
- `GET /state/{session_id}`
- `GET /health`

## Themes

**Theme 1: Multi-Agent Interactions** - theory-of-mind reasoning, asymmetric information, competitive dynamics, emergent strategic behavior.

**Theme 3.1: World Modeling** - partially observable environment, persistent state, multi-step decision making under real procurement constraints.

## Citation

Built on AgenticPay: Liu et al. (2026) arXiv:2602.06008
