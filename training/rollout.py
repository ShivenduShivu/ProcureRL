from contextlib import nullcontext
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

from agenticpay.openenv_adapter.procure_env_extended import ProcureEnvExtended

from .prompt_builder import format_for_trl


def _prepare_inputs(model, tokenizer, messages: List[Dict[str, str]]) -> Any:
    if torch is not None:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(input_ids, "to"):
            device = getattr(model, "device", "cpu")
            input_ids = input_ids.to(device)
        return input_ids

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _decode_prompt(tokenizer, prepared_inputs: Any) -> str:
    if isinstance(prepared_inputs, str):
        return prepared_inputs

    try:
        return tokenizer.decode(prepared_inputs[0], skip_special_tokens=True)
    except Exception:
        return str(prepared_inputs)


def _generate_action(
    model,
    tokenizer,
    prepared_inputs: Any,
    max_new_tokens: int,
    temperature: float,
) -> str:
    if torch is not None and not isinstance(prepared_inputs, str):
        no_grad = torch.no_grad()
    else:
        no_grad = nullcontext()

    with no_grad:
        output = model.generate(
            prepared_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )

    if isinstance(output, str):
        return output

    if torch is not None and not isinstance(prepared_inputs, str):
        try:
            new_tokens = output[0][prepared_inputs.shape[-1] :]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception:
            pass

    try:
        return tokenizer.decode(output, skip_special_tokens=True)
    except Exception:
        return str(output)


def run_episode(
    model,
    tokenizer,
    env: ProcureEnvExtended,
    difficulty: str = "easy",
    seed: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
) -> Dict[str, Any]:
    """
    Run one complete negotiation episode.
    Returns trajectory with prompts, responses, rewards, and breakdown.
    """
    obs, info = env.reset(seed=seed)
    done = False
    trajectory = {
        "prompts": [],
        "responses": [],
        "rewards": [],
        "reward_breakdowns": [],
        "info": [info],
    }

    while not done:
        messages = format_for_trl(obs)
        prepared_inputs = _prepare_inputs(model, tokenizer, messages)
        action_text = _generate_action(
            model,
            tokenizer,
            prepared_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        next_obs, reward, terminated, truncated, step_info = env.step(action_text)
        done = terminated or truncated

        trajectory["prompts"].append(_decode_prompt(tokenizer, prepared_inputs))
        trajectory["responses"].append(action_text)
        trajectory["rewards"].append(float(reward))
        breakdown = getattr(env, "_last_reward_breakdown", None)
        trajectory["reward_breakdowns"].append(breakdown.to_dict() if breakdown else {})
        trajectory["info"].append(step_info)
        obs = next_obs

    return trajectory


def collect_rollouts(
    model,
    tokenizer,
    n_episodes: int = 16,
    difficulty: str = "easy",
    base_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Collect n_episodes rollouts for one training step."""
    env = ProcureEnvExtended(difficulty=difficulty)
    trajectories = []
    for i in range(n_episodes):
        traj = run_episode(
            model,
            tokenizer,
            env,
            difficulty=difficulty,
            seed=base_seed + i,
        )
        trajectories.append(traj)
    return trajectories
