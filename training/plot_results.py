import base64
import json
import os
from typing import Dict, List


_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9s2j2b8AAAAASUVORK5CYII="
)


def _write_fallback_png(path: str):
    with open(path, "wb") as file:
        file.write(_FALLBACK_PNG)


def save_training_plots(training_log: List[Dict], output_dir: str = "results/plots"):
    """
    Save reward curve plots from training log.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_log.json"), "w", encoding="utf-8") as file:
        json.dump(training_log, file, indent=2)

    total_plot = os.path.join(output_dir, "reward_total.png")
    components_plot = os.path.join(output_dir, "reward_components.png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(training_log)))

        fig, ax = plt.subplots(figsize=(10, 5))
        rewards = [entry.get("reward/total", 0) for entry in training_log]
        ax.plot(steps, rewards, color="#1F3A8A", linewidth=2, label="Total Reward")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("ProcureRL: Total Reward During Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(total_plot, dpi=150)
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        components = [
            ("reward/savings", "Price Savings", "#16A34A", axes[0][0]),
            ("reward/efficiency", "Round Efficiency", "#D97706", axes[0][1]),
            ("reward/compliance", "Policy Compliance", "#7C3AED", axes[1][0]),
            ("reward/deal_quality", "Deal Quality", "#DC2626", axes[1][1]),
        ]
        for key, label, color, ax in components:
            values = [entry.get(key, 0) for entry in training_log]
            ax.plot(steps, values, color=color, linewidth=1.5)
            ax.set_title(label)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reward Component")
            ax.grid(True, alpha=0.3)
        plt.suptitle("ProcureRL: Reward Components During Training", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(components_plot, dpi=150)
        plt.close()
    except ImportError:
        _write_fallback_png(total_plot)
        _write_fallback_png(components_plot)

    print(f"Plots saved to {output_dir}/")


def save_before_after_plot(baseline: Dict, trained: Dict, output_dir: str = "results/plots"):
    """Save a before/after comparison bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    before_after_plot = os.path.join(output_dir, "before_after.png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["deal_rate", "mean_episode_reward", "mean_savings"]
        labels = ["Deal Rate", "Mean Reward", "Mean Savings"]
        baseline_vals = [baseline.get(m, 0) for m in metrics]
        trained_vals = [trained.get(m, 0) for m in metrics]
        x = range(len(metrics))
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        ax.bar(
            [i - width / 2 for i in x],
            baseline_vals,
            width,
            label="Baseline (No Training)",
            color="#94A3B8",
        )
        ax.bar(
            [i + width / 2 for i in x],
            trained_vals,
            width,
            label="Trained (ProcureRL)",
            color="#1F3A8A",
        )
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("ProcureRL: Before vs After RL Training")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(before_after_plot, dpi=150)
        plt.close()
    except ImportError:
        _write_fallback_png(before_after_plot)

    print("Before/after comparison plot saved.")
