import json
import os
import shutil

from training.evaluate import evaluate_model, print_evaluation_report
from training.plot_results import save_before_after_plot, save_training_plots
from training.prompt_builder import build_prompt, format_for_trl
from training.rollout import run_episode


SAMPLE_OBS = {
    "current_round": 2,
    "max_rounds": 10,
    "seller_last_price": 125.0,
    "buyer_max_price": 120.0,
    "market_signal_low": 100.0,
    "market_signal_high": 110.0,
    "competitor_signal": 108.0,
    "has_competitor": True,
    "conversation_history": [],
    "seller_offered_delivery_days": 25,
    "buyer_max_delivery_days": 21,
    "seller_offered_quality_tier": 2,
    "buyer_min_quality_tier": 2,
    "policy_budget_ceiling": 120.0,
}


class DummyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_tensors=None):
        parts = [f"{msg['role'].upper()}: {msg['content']}" for msg in messages]
        prompt = "\n".join(parts)
        if add_generation_prompt:
            prompt += "\nASSISTANT:"
        return prompt

    def decode(self, payload, skip_special_tokens=True):
        return payload if isinstance(payload, str) else str(payload)


class DummyModel:
    device = "cpu"

    def generate(self, inputs, max_new_tokens=256, temperature=0.8, do_sample=True, pad_token_id=None):
        return "I can offer $50 today. <BUYER_PRICE>50</BUYER_PRICE>"


def test_prompt_builder_includes_budget():
    prompt = build_prompt(SAMPLE_OBS)
    assert "120" in prompt


def test_prompt_builder_includes_seller_price():
    prompt = build_prompt(SAMPLE_OBS)
    assert "125" in prompt


def test_prompt_builder_includes_competitor():
    prompt = build_prompt(SAMPLE_OBS)
    assert "108" in prompt


def test_format_for_trl_returns_messages_list():
    messages = format_for_trl(SAMPLE_OBS)
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_plot_functions_create_pngs():
    tmpdir = os.path.join("results", "test_plots")
    try:
        os.makedirs(tmpdir, exist_ok=True)
        log = [
            {
                "reward/total": 0.1 * i,
                "reward/savings": 0.05 * i,
                "reward/efficiency": 0.1,
                "reward/compliance": 1.0,
                "reward/deal_quality": 0.5,
            }
            for i in range(20)
        ]
        save_training_plots(log, output_dir=tmpdir)
        before = {"deal_rate": 0.3, "mean_episode_reward": 0.1, "mean_savings": 0.05}
        after = {"deal_rate": 0.7, "mean_episode_reward": 0.3, "mean_savings": 0.15}
        save_before_after_plot(before, after, output_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "reward_total.png"))
        assert os.path.exists(os.path.join(tmpdir, "reward_components.png"))
        assert os.path.exists(os.path.join(tmpdir, "before_after.png"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_notebook_file_exists_and_is_valid_json():
    notebook_path = "notebooks/ProcureRL_Training.ipynb"
    assert os.path.exists(notebook_path)
    with open(notebook_path, "r", encoding="utf-8") as file:
        notebook = json.load(file)
    assert "cells" in notebook
    notebook_text = json.dumps(notebook)
    assert "load_in_4bit" in notebook_text
    assert "save_pretrained_merged" in notebook_text


def test_run_episode_executes_multiple_steps():
    from agenticpay.openenv_adapter.procure_env_extended import ProcureEnvExtended

    env = ProcureEnvExtended(difficulty="easy")
    trajectory = run_episode(DummyModel(), DummyTokenizer(), env, seed=7)
    assert len(trajectory["rewards"]) >= 3
    assert len(trajectory["responses"]) == len(trajectory["rewards"])


def test_evaluate_model_runs_with_dummy_model():
    metrics = evaluate_model(DummyModel(), DummyTokenizer(), n_episodes=5, difficulty="easy")
    print_evaluation_report(metrics, label="Dummy")
    assert metrics["n_episodes"] == 5
    assert "mean_episode_reward" in metrics
