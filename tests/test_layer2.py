from agenticpay.openenv_adapter import ProcureEnv
from agenticpay.openenv_adapter.procure_env_extended import ProcureEnvExtended
from agenticpay.openenv_adapter.scenario_generator import generate_scenario


def test_extended_env_reset():
    env = ProcureEnvExtended(difficulty="easy")
    obs, info = env.reset(seed=42)
    assert "seller_offered_delivery_days" in obs
    assert "seller_offered_quality_tier" in obs
    assert "competitor_signal" in obs
    assert "policy_budget_ceiling" in obs


def test_extended_step_returns_five_tuple():
    env = ProcureEnvExtended()
    env.reset(seed=42)
    result = env.step("Offer $110. <BUYER_PRICE>110</BUYER_PRICE>")
    assert len(result) == 5


def test_layer1_regression_still_works():
    env = ProcureEnv()
    obs, info = env.reset()
    assert "seller_last_price" in obs


def test_scenario_generator_produces_valid_scenarios():
    easy = generate_scenario(difficulty="easy", seed=1)
    hard = generate_scenario(difficulty="hard", seed=1)
    for difficulty in ["easy", "medium", "hard"]:
        scenario = generate_scenario(difficulty=difficulty, seed=1)
        assert scenario.seller_min_price < scenario.buyer_max_price
        assert scenario.initial_seller_price > scenario.seller_min_price
        assert scenario.difficulty == difficulty
    assert hard.max_rounds < easy.max_rounds


def test_full_extended_episode():
    env = ProcureEnvExtended(difficulty="easy")
    obs, _ = env.reset(seed=99)
    done = False
    steps = 0
    while not done and steps < 20:
        price = obs["seller_last_price"] - 2
        action = f"Offer ${price:.2f}. <BUYER_PRICE>{price:.2f}</BUYER_PRICE>"
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done


def test_policy_constraint_violation_penalized():
    env = ProcureEnvExtended()
    obs, _ = env.reset(seed=1)
    ceiling = obs["policy_budget_ceiling"]
    over_budget = ceiling + 20
    _, reward, terminated, truncated, info = env.step(
        f"I agree to ${over_budget:.2f}. <BUYER_PRICE>{over_budget:.2f}</BUYER_PRICE>"
    )
    if terminated:
        assert reward < 0, "Budget ceiling violation must return negative reward"
