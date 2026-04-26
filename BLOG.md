---
title: "We Tried to Teach an AI to Negotiate. It Was Humbling."
emoji: 🤝
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# We Tried to Teach an AI to Negotiate. It Was Humbling.

*By Shivendu — Team 404 NOT FOUNDERS | Meta OpenEnv Hackathon 2026*

---

I'll be honest with you. When we started ProcureRL, we thought
"how hard can it be to teach an AI to haggle?"

Very hard. The answer is very hard.

---

## The Problem Nobody Talks About

Everyone's training LLMs to write code, solve math, summarize
documents. Cool. But here's a capability gap that's worth billions
of dollars and nobody's touching it in RL research:

**Procurement negotiation.**

Every company — from a two-person startup buying cloud credits to
a manufacturer sourcing steel — has someone whose entire job is
sitting across from a supplier and fighting for a better deal.
Price, delivery timeline, quality guarantees. All at once. Under
pressure. With incomplete information about what the other side
will actually accept.

Ask any current LLM to do this and watch what happens. It either
accepts the first offer like an eager intern on day one, or it
makes an absurdly low bid that no supplier would ever take
seriously.

There's no middle ground. No strategy. No reading the room.

We wanted to fix that.

---

## What We Actually Built

**ProcureRL** is a reinforcement learning environment where an
LLM buyer agent learns to negotiate procurement deals. Built on
top of [AgenticPay](https://arxiv.org/pdf/2602.06008) and wrapped
in the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) spec.

Here's what makes it interesting:

The buyer agent operates in a genuinely asymmetric information
setting. It knows its own budget ceiling — but the seller doesn't.
The seller knows its own cost floor — but the buyer doesn't. Both
are trying to get the best deal possible, and neither has the full
picture. Sound familiar? That's because this is how real
procurement works.

The negotiation covers three dimensions simultaneously:
- 💰 **Price** — obviously
- 📦 **Delivery timeline** — faster costs more
- ⭐ **Quality tier** — better quality, higher ask

Plus: the buyer gets a noisy market price signal (what's the going
rate?) and sometimes a competitor quote (there's another supplier,
and they're cheaper). These are real negotiation levers. Using
them well is a learnable skill.

The seller is a scripted policy — linear concession strategy,
starts high, drops slowly each round. We did this intentionally.
If you're training one agent, you want a stable opponent.
Otherwise you're training two agents to confuse each other, which
is a different and harder problem.

---

## The Reward Function: Where We Suffered

Here's the part nobody tells you about RL environments: designing
the reward function is where you actually find out how hard your
problem is.

**Attempt 1:** Reward +1 if deal, 0 if no deal.
Clean. Simple. Completely useless. GRPO couldn't tell the
difference between a brilliant opening offer and a terrible one
because neither closes the deal in round one.

**Attempt 2:** Full episode rollout.
The agent writes one message, a scripted buyer finishes the rest
of the episode, we evaluate the outcome. Seemed smart. Was not
smart. Deal rate collapsed from 86% to 16%. The agent learned to
write a message that worked well when someone else finished the
negotiation. Completely different from what we were trying to
train.

**Attempt 3 (the one that actually worked):** Shaped intermediate
reward.

We built a reward function that scores the strategic positioning
of each offer relative to the specific scenario's price scale.
Not just "did you close?" but "was this a smart opening given
what you know?"

The key insight: an offer of $1900 is brilliant in a $2320-budget
scenario but absurd in a $5750-budget scenario. The reward
function has to know the difference. Ours does now.

**Reward components:**
- **Positioning score** — how strategically placed is this offer?
- **Efficiency score** — fewer rounds = better
- **Format compliance** — did you include the price tag?
- **Hard constraint** — exceed budget ceiling = -1.0, no exceptions

After three training runs, two runtime crashes, one catastrophic
reward function redesign, and more debugging than we'd like to
admit — the model learned to eliminate budget ceiling violations
entirely.

**Constraint violation rate: 13.3% → 0.0%**

Is that the most dramatic improvement? No. Is it a real,
meaningful behavioral change that proves the agent learned
something? Yes.

---

## The Technical Stack

| Component | Choice |
|---|---|
| Model | Qwen2.5-3B-Instruct |
| Quantization | 4-bit QLoRA via Unsloth |
| Training | GRPO with TRL |
| Environment | OpenEnv-compliant FastAPI |
| Deployment | HuggingFace Spaces |
| Tests | 44 tests across 5 layers |

---

## Training Evidence

The model trained for 150 steps with non-zero reward variance
throughout — meaning GRPO had real signal to learn from across
the entire run.

The environment improved significantly over time, but the cleanest
validated training result came from an earlier run before later
ablations and Colab instability complicated evaluation.

Key metrics after training:
- Deal rate maintained at ~87%
- Budget constraint violations eliminated (13.3% → 0%)
- Mean savings: 15% below budget on closed deals
- Average rounds to close: 4.4 out of 10

![Training Reward](results/plots/reward_total.png)
*Reward during training — non-zero variance throughout*

![Before vs After](results/plots/before_after.png)
*Baseline vs trained model comparison*

---

## Try It Yourself

The environment is live and runnable:

```python
import requests

# Start a negotiation episode
r = requests.post(
    'https://starwarrior24x7-procurerl.hf.space/reset',
    json={'difficulty': 'easy', 'seed': 42}
)
session_id = r.json()['session_id']
obs = r.json()['observation']

print(f"Seller asking: ${obs['seller_last_price']}")
print(f"Your budget: ${obs['buyer_max_price']}")
print(f"Market range: ${obs['market_signal_low']} - ${obs['market_signal_high']}")

# Make your move
r2 = requests.post(
    'https://starwarrior24x7-procurerl.hf.space/step',
    json={
        'session_id': session_id,
        'action': "Market data supports a lower price. I offer $1850. <BUYER_PRICE>1850</BUYER_PRICE>"
    }
)
print(f"Reward: {r2.json()['reward']}")
print(f"Seller response: {r2.json()['observation']['conversation_history'][-1]}")
```

**Links:**
- 🚀 [Live Environment](https://starwarrior24x7-procurerl.hf.space)
- 📓 [Training Notebook (Colab)](https://colab.research.google.com/drive/1E3w2Uac9HYaPov4_lOiFdqVxk7lMlb1e?usp=sharing)
- 💻 [GitHub Repository](https://github.com/ShivenduShivu/ProcureRL)

---

## What's Next

Honestly? A lot.

Seller personality diversity (rigid vs gradual conceder vs
market-driven). Multi-stage curriculum. Better credit assignment
across turns. Structured action space.

ProcureRL v1 is a proof of concept that this domain is worth
training on. We think it is. Procurement is a trillion-dollar
domain where AI assistance is still basically non-existent at
the negotiation layer.

Someone should fix that. Might as well be us.

---

*Built by Shivendu — Team 404 NOT FOUNDERS*
*Meta OpenEnv Hackathon 2026*

---
