# ProcureRL — Project Report

## What Problem Are We Solving
Procurement negotiation looks simple on the surface, but it is difficult for language models to do well. A good negotiator does not just keep asking for a lower price. They have to balance price, delivery speed, quality, market conditions, and internal policy limits, all while reasoning under incomplete information. The buyer does not know the seller's true floor price, and the seller does not know the buyer's full budget ceiling. That makes the task strategic rather than purely conversational.

In the real world, bad negotiation is expensive. If an AI buyer accepts a bad price, takes poor delivery terms, or violates a policy budget ceiling, the result is not just a weak conversation outcome. It can lead to direct financial loss, compliance risk, and operational delays. Large organizations make procurement decisions constantly, so even small errors repeated at scale become meaningful.

The gap we are filling is that existing systems mostly benchmark or simulate negotiation, but they do not provide a clean reinforcement learning environment focused on procurement-specific behavior. ProcureRL turns procurement negotiation into something we can train on, measure, and improve systematically.

## What We Built
We built a negotiation environment where a buyer agent learns to negotiate against a fixed seller. On each turn, the buyer sees the seller's current offer, a noisy market price range, optional competitor pressure, and in the extended environment, delivery and quality terms as well. It also sees the recent conversation history, so each decision happens in context rather than in isolation.

The buyer's job is to respond with a natural-language negotiation message that includes a structured price offer. The environment then applies the seller's scripted response, updates the negotiation state, and computes a reward. That reward does not just reflect whether a deal happened. It reflects whether the deal was cheap enough, fast enough, policy-compliant, and good enough across multiple procurement dimensions.

An easy analogy is a flight simulator for negotiation. The model is the pilot. The environment provides the cockpit instruments, the weather, and the constraints. The agent makes a move, the simulator reacts, and the system scores how well the decision balanced all the goals at once.

## What Makes It Original
First, ProcureRL is built as a trainable procurement environment rather than only a negotiation benchmark. That changes the focus from measuring model behavior to improving it through RL.

Second, the environment is multi-variable. It does not only negotiate price. It also includes delivery days and quality tier, which makes it closer to real procurement trade-offs than many price-only negotiation tasks.

Third, it introduces procurement-specific signals and constraints. The buyer sees noisy market information and optional competitor pressure, while a hard internal budget ceiling is enforced through reward logic. That makes the task grounded in enterprise decision-making rather than abstract bargaining.

Fourth, the project includes an end-to-end path from environment to deployment: OpenEnv-style interaction, centralized reward logic, GRPO training infrastructure, a FastAPI server, and Docker packaging for Spaces deployment. That combination makes it usable as both a research artifact and a live demo system.

## How The System Works
The environment creates a procurement scenario and shows the buyer the current situation. The agent reads that state and writes its next negotiation message, including a price offer. The environment sends that message to a scripted seller, updates the conversation, checks whether a deal happened, and calculates a reward based on the outcome. Those rewards are then fed into training so the buyer model gradually learns which kinds of negotiation moves lead to better deals and fewer bad decisions.

## The Training Approach
We use GRPO through TRL, with Unsloth handling efficient low-memory fine-tuning of Qwen2.5-3B-Instruct in 4-bit QLoRA form. In plain terms, this gives us a practical way to train a reasonably capable model on a single Colab T4 GPU without needing a large research cluster.

This stack was chosen because it is realistic for a hackathon setting. TRL gives us the RL trainer, Unsloth makes fine-tuning small enough to fit within commodity GPU limits, and Qwen2.5-3B is strong enough to produce strategic language while still being trainable under budget.

We train only the buyer. That is deliberate. If both buyer and seller learn at the same time, the problem becomes much less stable and much harder to interpret. By keeping the seller scripted, we make the training signal more reproducible and easier to compare before and after training. This also lets us focus the optimization on the side that matters most for our use case: the procurement agent.

## Reward Design
The reward has four main parts. Savings measures how much the buyer preserved relative to its budget ceiling, normalized against the feasible negotiation range. Efficiency rewards closing the negotiation in fewer rounds, because long negotiations consume time and attention in real procurement work.

Compliance measures whether the buyer stayed within procurement rules. In the current design, that mainly means respecting the policy budget ceiling, but the structure is ready for future extensions such as disclosure or formatting rules. Deal quality captures the non-price dimensions of the agreement, especially delivery timing and quality tier in the extended environment.

Hard constraint violations are treated differently from ordinary suboptimal behavior. If the agreed price exceeds the budget ceiling, the reward becomes `-1.0` regardless of any other positive components. That is intentional. In real procurement, some mistakes are not "slightly bad"; they are unacceptable. The hard penalty prevents the model from learning to trade compliance away in exchange for short-term reward elsewhere.

## Curriculum Learning
ProcureRL uses an easy, medium, and hard progression. In the easy setting, the number of rounds is larger, the tolerance is more forgiving, and the negotiation gap is more manageable. As difficulty increases, the environment allows fewer rounds, tighter agreement conditions, and more competitive settings.

We do not start with the hardest setting because early RL training is fragile. If the model almost never gets a useful reward at the beginning, it has very little signal to learn from. The curriculum gives the agent simpler cases first, so it can discover the basic structure of successful negotiation before being asked to handle tighter constraints and harder scenarios.

## Current Status
Layer 1 is complete. The OpenEnv-compatible adapter, typed observation and action objects, scripted seller, manifest file, and verification tests are all implemented.

Layer 2 is complete. The procurement-specific extension with multi-variable negotiation, scenario generation, and competitive pressure signals is implemented and tested.

Layer 3 is complete. A centralized reward engine with multi-component breakdown, hard constraints, and anti-gaming logic now drives both the base and extended environments.

Layer 4 is complete from the code and infrastructure side. The TRL plus GRPO training notebook, rollout code, evaluation helpers, and plot generation utilities are implemented. Training is currently in progress on a Colab T4 GPU.

Layer 5 is complete from the deployment side. The FastAPI server, Dockerfile, local API tests, deployment-oriented README, and Spaces-compatible runtime path are implemented. At the repository level, the full automated test suite currently passes at 47 out of 47 tests.

## Strengths of This Approach
One clear strength is that the project is end-to-end. It is not only an environment, not only a trainer, and not only a demo server. The pieces connect cleanly from negotiation state to reward to RL training to deployment.

A second strength is that the environment reflects real procurement trade-offs more closely than a pure price-negotiation task. Delivery time, quality tier, market signals, competitor pressure, and hard budget ceilings make the training objective more realistic.

A third strength is reproducibility. The scripted seller and centralized reward engine make it much easier to compare model behavior across runs, debug failure modes, and reason about why a reward changed.

A fourth strength is practicality. The stack is deliberately chosen to run on modest hardware, which makes the project easier to reproduce and demonstrate in a hackathon setting.

## Known Limitations and What We'd Do With More Time
The biggest current limitation is that the final training evidence is not yet complete inside the repository. The code path for training, evaluation, and plot generation exists, but the final Colab-generated metrics and plots still need to be committed after the long run finishes. That is a process limitation rather than an architecture limitation, but it matters for submission readiness.

The second limitation is that the seller is scripted rather than learned. This is the right choice for stability and interpretability right now, but it means the negotiation dynamics are still narrower than a fully adaptive marketplace. With more time, we would explore stronger seller policies or multi-policy evaluation while preserving training stability.

The third limitation is that the current deployment path is optimized for CPU-only inference and environment serving, not for running the training loop inside the deployment target. That is appropriate for HuggingFace Spaces, but a production-grade research setup would likely separate training infrastructure, inference serving, and benchmark evaluation more cleanly.
