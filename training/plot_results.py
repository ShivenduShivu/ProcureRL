import json
import os
from typing import Dict, List, Optional


def _get_steps(training_log: List[Dict]) -> List[float]:
    if not training_log:
        return []
    if "step" in training_log[0]:
        return [entry.get("step", index + 1) for index, entry in enumerate(training_log)]
    return list(range(1, len(training_log) + 1))


def _normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.5
    return (value - min_value) / (max_value - min_value)


def _draw_text(draw, position, text, fill, font):
    draw.text(position, text, fill=fill, font=font)


def _draw_axes(draw, box, title, x_label, y_label, font, title_font):
    left, top, right, bottom = box
    axis_left = left + 70
    axis_top = top + 35
    axis_right = right - 25
    axis_bottom = bottom - 55

    draw.line((axis_left, axis_top, axis_left, axis_bottom), fill="#334155", width=2)
    draw.line((axis_left, axis_bottom, axis_right, axis_bottom), fill="#334155", width=2)
    _draw_text(draw, (left + 10, top + 5), title, fill="#0f172a", font=title_font)
    _draw_text(draw, (axis_left + (axis_right - axis_left) // 2 - 45, bottom - 35), x_label, fill="#334155", font=font)
    _draw_text(draw, (left + 5, axis_top - 10), y_label, fill="#334155", font=font)

    return axis_left, axis_top, axis_right, axis_bottom


def _draw_line_chart(draw, box, xs, ys, title, x_label, y_label, line_color, font, title_font, y_min=None, y_max=None):
    axis_left, axis_top, axis_right, axis_bottom = _draw_axes(
        draw,
        box,
        title,
        x_label,
        y_label,
        font,
        title_font,
    )
    if not xs or not ys:
        return

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys) if y_min is None else y_min
    max_y = max(ys) if y_max is None else y_max
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0

    for index in range(5):
        tick_value = min_y + (max_y - min_y) * (4 - index) / 4
        y = axis_top + (axis_bottom - axis_top) * index / 4
        draw.line((axis_left, y, axis_right, y), fill="#e2e8f0", width=1)
        _draw_text(draw, (axis_left - 58, y - 7), f"{tick_value:.2f}", fill="#475569", font=font)

    points = []
    for x, y in zip(xs, ys):
        x_norm = _normalize(x, min_x, max_x) if max_x != min_x else 0.5
        y_norm = _normalize(y, min_y, max_y)
        px = axis_left + x_norm * (axis_right - axis_left)
        py = axis_bottom - y_norm * (axis_bottom - axis_top)
        points.append((px, py))

    if len(points) >= 2:
        draw.line(points, fill=line_color, width=3)

    for point, x_value, y_value in zip(points, xs, ys):
        px, py = point
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=line_color, outline=line_color)
        _draw_text(draw, (px - 10, axis_bottom + 8), str(int(x_value)), fill="#475569", font=font)
        _draw_text(draw, (px - 12, py - 20), f"{y_value:.2f}", fill=line_color, font=font)


def _draw_grouped_bar_chart(draw, box, categories, baseline_vals, trained_vals, title, y_label, font, title_font):
    axis_left, axis_top, axis_right, axis_bottom = _draw_axes(
        draw,
        box,
        title,
        "Metric",
        y_label,
        font,
        title_font,
    )

    all_values = baseline_vals + trained_vals
    min_y = min(0.0, min(all_values))
    max_y = max(all_values + [0.0])
    if min_y == max_y:
        max_y = min_y + 1.0

    for index in range(5):
        tick_value = min_y + (max_y - min_y) * (4 - index) / 4
        y = axis_top + (axis_bottom - axis_top) * index / 4
        draw.line((axis_left, y, axis_right, y), fill="#e2e8f0", width=1)
        _draw_text(draw, (axis_left - 60, y - 7), f"{tick_value:.2f}", fill="#475569", font=font)

    span = (axis_right - axis_left) / max(len(categories), 1)
    zero_y = axis_bottom - _normalize(0.0, min_y, max_y) * (axis_bottom - axis_top)

    for index, category in enumerate(categories):
        center = axis_left + span * (index + 0.5)
        bar_width = span * 0.22

        baseline_height = axis_bottom - _normalize(baseline_vals[index], min_y, max_y) * (axis_bottom - axis_top)
        trained_height = axis_bottom - _normalize(trained_vals[index], min_y, max_y) * (axis_bottom - axis_top)

        draw.rectangle(
            (center - bar_width - 6, min(zero_y, baseline_height), center - 6, max(zero_y, baseline_height)),
            fill="#94A3B8",
            outline="#64748B",
        )
        draw.rectangle(
            (center + 6, min(zero_y, trained_height), center + bar_width + 6, max(zero_y, trained_height)),
            fill="#1F3A8A",
            outline="#1D4ED8",
        )

        _draw_text(draw, (center - 36, axis_bottom + 8), category, fill="#475569", font=font)
        _draw_text(draw, (center - bar_width - 12, baseline_height - 18 if baseline_height < zero_y else baseline_height + 4), f"{baseline_vals[index]:.3f}", fill="#64748B", font=font)
        _draw_text(draw, (center + 2, trained_height - 18 if trained_height < zero_y else trained_height + 4), f"{trained_vals[index]:.3f}", fill="#1D4ED8", font=font)

    _draw_text(draw, (axis_right - 220, axis_top + 8), "Baseline", fill="#64748B", font=font)
    draw.rectangle((axis_right - 245, axis_top + 10, axis_right - 228, axis_top + 24), fill="#94A3B8", outline="#64748B")
    _draw_text(draw, (axis_right - 120, axis_top + 8), "Trained", fill="#1D4ED8", font=font)
    draw.rectangle((axis_right - 145, axis_top + 10, axis_right - 128, axis_top + 24), fill="#1F3A8A", outline="#1D4ED8")


def _save_with_pillow(training_log: List[Dict], output_dir: str, baseline: Optional[Dict], trained: Optional[Dict]):
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    steps = _get_steps(training_log)
    rewards = [entry.get("reward/total", 0.0) for entry in training_log]

    total_image = Image.new("RGB", (1200, 700), "white")
    total_draw = ImageDraw.Draw(total_image)
    _draw_line_chart(
        total_draw,
        (40, 30, 1160, 660),
        steps,
        rewards,
        "ProcureRL: Mean Episode Reward Across 60 Training Steps",
        "Training Step",
        "Mean Episode Reward",
        "#1F3A8A",
        font,
        title_font,
    )
    total_image.save(os.path.join(output_dir, "reward_total.png"))

    component_image = Image.new("RGB", (1200, 900), "white")
    component_draw = ImageDraw.Draw(component_image)
    _draw_line_chart(
        component_draw,
        (40, 20, 1160, 430),
        steps,
        rewards,
        "Reward Trend During Training",
        "Training Step",
        "Mean Episode Reward",
        "#2563EB",
        font,
        title_font,
    )

    if baseline and trained:
        categories = ["Baseline", "Trained"]
        violation_vals = [
            baseline.get("constraint_violation_rate", 0.0),
            trained.get("constraint_violation_rate", 0.0),
        ]
        _draw_grouped_bar_chart(
            component_draw,
            (40, 450, 1160, 870),
            ["Constraint\nViolation"],
            [violation_vals[0]],
            [violation_vals[1]],
            "Constraint Violation Rate Drops to Zero After Training",
            "Rate",
            font,
            title_font,
        )
    component_image.save(os.path.join(output_dir, "reward_components.png"))

    comparison_image = Image.new("RGB", (1200, 760), "white")
    comparison_draw = ImageDraw.Draw(comparison_image)
    baseline_vals = [
        baseline.get("deal_rate", 0.0) if baseline else 0.0,
        baseline.get("mean_episode_reward", 0.0) if baseline else 0.0,
        baseline.get("constraint_violation_rate", 0.0) if baseline else 0.0,
    ]
    trained_vals = [
        trained.get("deal_rate", 0.0) if trained else 0.0,
        trained.get("mean_episode_reward", 0.0) if trained else 0.0,
        trained.get("constraint_violation_rate", 0.0) if trained else 0.0,
    ]
    _draw_grouped_bar_chart(
        comparison_draw,
        (40, 30, 1160, 720),
        ["Deal Rate", "Mean Reward", "Constraint Viol."],
        baseline_vals,
        trained_vals,
        "ProcureRL: Baseline vs Trained Policy",
        "Value",
        font,
        title_font,
    )
    comparison_image.save(os.path.join(output_dir, "before_after.png"))


def _save_before_after_with_pillow(output_dir: str, baseline: Dict, trained: Dict):
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    comparison_image = Image.new("RGB", (1200, 760), "white")
    comparison_draw = ImageDraw.Draw(comparison_image)
    baseline_vals = [
        baseline.get("deal_rate", 0.0),
        baseline.get("mean_episode_reward", 0.0),
        baseline.get("constraint_violation_rate", 0.0),
    ]
    trained_vals = [
        trained.get("deal_rate", 0.0),
        trained.get("mean_episode_reward", 0.0),
        trained.get("constraint_violation_rate", 0.0),
    ]
    _draw_grouped_bar_chart(
        comparison_draw,
        (40, 30, 1160, 720),
        ["Deal Rate", "Mean Reward", "Constraint Viol."],
        baseline_vals,
        trained_vals,
        "ProcureRL: Baseline vs Trained Policy",
        "Value",
        font,
        title_font,
    )
    comparison_image.save(os.path.join(output_dir, "before_after.png"))


def save_training_plots(
    training_log: List[Dict],
    output_dir: str = "results/plots",
    baseline: Optional[Dict] = None,
    trained: Optional[Dict] = None,
):
    """
    Save training plots as PNG files in the requested output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_log.json"), "w", encoding="utf-8") as file:
        json.dump(training_log, file, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = _get_steps(training_log)
        rewards = [entry.get("reward/total", 0.0) for entry in training_log]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rewards, color="#1F3A8A", linewidth=2, marker="o", label="Mean Episode Reward")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("ProcureRL: Mean Episode Reward Across 60 Training Steps")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_total.png"), dpi=150)
        plt.close()

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(steps, rewards, color="#2563EB", linewidth=2, marker="o")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Mean Episode Reward")
        axes[0].set_title("Reward Trend During Training")
        axes[0].grid(True, alpha=0.3)

        if baseline and trained:
            stages = ["Baseline", "Trained"]
            violation_values = [
                baseline.get("constraint_violation_rate", 0.0),
                trained.get("constraint_violation_rate", 0.0),
            ]
            axes[1].bar(stages, violation_values, color=["#94A3B8", "#1F3A8A"])
            axes[1].set_xlabel("Training Step")
            axes[1].set_ylabel("Mean Episode Reward")
            axes[1].set_title("Constraint Violation Rate Drops to Zero After Training")
            axes[1].grid(True, alpha=0.3, axis="y")
            for stage, value in zip(stages, violation_values):
                axes[1].text(stage, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
        else:
            axes[1].plot(steps, [entry.get("metric/constraint_violated", 0.0) for entry in training_log], color="#DC2626", linewidth=2, marker="o")
            axes[1].set_xlabel("Training Step")
            axes[1].set_ylabel("Mean Episode Reward")
            axes[1].set_title("Constraint Violation Signal During Training")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_components.png"), dpi=150)
        plt.close()
    except ImportError:
        _save_with_pillow(training_log, output_dir, baseline, trained)

    print(f"Plots saved to {output_dir}/")


def save_before_after_plot(baseline: Dict, trained: Dict, output_dir: str = "results/plots"):
    """Save a before/after comparison PNG in the requested output directory."""
    os.makedirs(output_dir, exist_ok=True)
    before_after_plot = os.path.join(output_dir, "before_after.png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["deal_rate", "mean_episode_reward", "constraint_violation_rate"]
        labels = ["Deal Rate", "Mean Reward", "Constraint Violations"]
        baseline_vals = [baseline.get(metric, 0.0) for metric in metrics]
        trained_vals = [trained.get(metric, 0.0) for metric in metrics]

        x = range(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([i - width / 2 for i in x], baseline_vals, width, label="Baseline", color="#94A3B8")
        ax.bar([i + width / 2 for i in x], trained_vals, width, label="Trained", color="#1F3A8A")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("ProcureRL: Baseline vs Trained Policy")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(before_after_plot, dpi=150)
        plt.close()
    except ImportError:
        _save_before_after_with_pillow(output_dir, baseline, trained)

    print("Before/after comparison plot saved.")
