from agenticpay.openenv_adapter import BuyerAction, ProcureEnv


def test_env_reset_returns_correct_keys():
    env = ProcureEnv()
    obs, info = env.reset()
    required_keys = [
        "current_round",
        "max_rounds",
        "seller_last_price",
        "buyer_max_price",
        "market_signal_low",
        "market_signal_high",
        "conversation_history",
        "deal_reached",
        "episode_over",
    ]
    for key in required_keys:
        assert key in obs, f"Missing key in observation: {key}"


def test_env_step_returns_five_tuple():
    env = ProcureEnv()
    env.reset()
    result = env.step("I can offer $115. <BUYER_PRICE>115</BUYER_PRICE>")
    assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"


def test_reward_is_float():
    env = ProcureEnv()
    env.reset()
    _, reward, _, _, _ = env.step("Offer $110. <BUYER_PRICE>110</BUYER_PRICE>")
    assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"


def test_constraint_violation_gives_negative_reward():
    env = ProcureEnv()
    env.reset()
    _, reward, terminated, _, _ = env.step("I agree to pay $130. <BUYER_PRICE>130</BUYER_PRICE>")
    if terminated:
        assert reward < 0, "Constraint violation must give negative reward"


def test_episode_ends_on_max_rounds():
    env = ProcureEnv({"max_rounds": 3})
    env.reset()
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step("No deal. <BUYER_PRICE>50</BUYER_PRICE>")
    assert truncated or terminated, "Episode must end by max_rounds"


def test_action_parser_extracts_price():
    action = BuyerAction.from_text("I offer $105. <BUYER_PRICE>105.0</BUYER_PRICE>")
    assert action.offered_price == 105.0


def test_action_parser_handles_no_price():
    action = BuyerAction.from_text("I need more time to consider.")
    assert action.offered_price is None


def test_full_episode_completes():
    env = ProcureEnv({"max_rounds": 5})
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        price = obs["seller_last_price"] - 2.0
        action = f"I can offer ${price:.2f}. <BUYER_PRICE>{price:.2f}</BUYER_PRICE>"
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps > 0
    assert "status" in info


def test_state_method_returns_dict():
    env = ProcureEnv()
    env.reset()
    state = env.state()
    assert isinstance(state, dict)
    assert "current_round" in state


def test_step_after_done_raises_runtime_error():
    env = ProcureEnv({"max_rounds": 1})
    env.reset()
    env.step("No deal. <BUYER_PRICE>50</BUYER_PRICE>")
    try:
        env.step("Second step should fail.")
        assert False, "Expected RuntimeError after episode end"
    except RuntimeError:
        assert True
