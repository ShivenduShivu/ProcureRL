import statistics
from typing import Dict

from agenticpay.openenv_adapter.procure_env_extended import ProcureEnvExtended

from .rollout import run_episode


def evaluate_model(
    model,
    tokenizer,
    n_episodes: int = 50,
    difficulty: str = "easy",
    base_seed: int = 10000,
) -> Dict[str, float]:
    """
    Evaluate model over n_episodes. Returns aggregated metrics.
    Call this before and after training to show improvement.
    """
    env = ProcureEnvExtended(difficulty=difficulty)
    all_rewards = []
    all_savings = []
    deal_count = 0
    constraint_violations = 0
    total_rounds = []

    for i in range(n_episodes):
        traj = run_episode(model, tokenizer, env, seed=base_seed + i, temperature=0.3)
        episode_reward = sum(traj["rewards"])
        all_rewards.append(episode_reward)
        total_rounds.append(len(traj["rewards"]))

        terminal_breakdown = traj["reward_breakdowns"][-1] if traj["reward_breakdowns"] else {}
        if terminal_breakdown.get("metric/deal_reached", 0) > 0:
            deal_count += 1
        if terminal_breakdown.get("metric/constraint_violated", 0) > 0:
            constraint_violations += 1

        savings = terminal_breakdown.get("reward/savings", 0)
        if savings > 0:
            all_savings.append(savings)

    return {
        "mean_episode_reward": statistics.mean(all_rewards) if all_rewards else 0.0,
        "median_episode_reward": statistics.median(all_rewards) if all_rewards else 0.0,
        "deal_rate": deal_count / n_episodes if n_episodes else 0.0,
        "constraint_violation_rate": constraint_violations / n_episodes if n_episodes else 0.0,
        "mean_rounds": statistics.mean(total_rounds) if total_rounds else 0.0,
        "mean_savings": statistics.mean(all_savings) if all_savings else 0.0,
        "n_episodes": n_episodes,
    }


def print_evaluation_report(metrics: Dict[str, float], label: str = "Model"):
    line = "=" * 50
    print(f"\n{line}")
    print(f"  EVALUATION REPORT: {label}")
    print(line)
    print(f"  Deal rate:              {metrics['deal_rate']:.1%}")
    print(f"  Mean episode reward:    {metrics['mean_episode_reward']:.4f}")
    print(f"  Mean savings:           {metrics['mean_savings']:.4f}")
    print(f"  Mean rounds to close:   {metrics['mean_rounds']:.1f}")
    print(f"  Constraint violations:  {metrics['constraint_violation_rate']:.1%}")
    print(f"{line}\n")
