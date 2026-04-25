from agenticpay.envs.reward_engine import ProcureRewardEngine, reward_engine


def test_deal_at_floor_gives_max_savings():
    reward = reward_engine.compute(
        deal_reached=True,
        agreed_price=95.0,
        buyer_max_price=120.0,
        seller_min_price=95.0,
        current_round=2,
        max_rounds=10,
        timed_out=False,
    )
    assert reward.r_savings == 1.0
    assert reward.deal_reached is True
    assert reward.constraint_violated is False


def test_deal_at_ceiling_gives_zero_savings():
    reward = reward_engine.compute(
        deal_reached=True,
        agreed_price=120.0,
        buyer_max_price=120.0,
        seller_min_price=95.0,
        current_round=5,
        max_rounds=10,
        timed_out=False,
    )
    assert reward.r_savings == 0.0


def test_constraint_violation_overrides_all():
    reward = reward_engine.compute(
        deal_reached=True,
        agreed_price=130.0,
        buyer_max_price=120.0,
        seller_min_price=95.0,
        current_round=1,
        max_rounds=10,
        timed_out=False,
    )
    assert reward.total == -1.0
    assert reward.constraint_violated is True


def test_timeout_gives_negative_reward():
    reward = reward_engine.compute(
        deal_reached=False,
        agreed_price=None,
        buyer_max_price=120.0,
        seller_min_price=95.0,
        current_round=10,
        max_rounds=10,
        timed_out=True,
    )
    assert reward.total < 0
    assert reward.deal_reached is False


def test_reward_always_in_range():
    import random

    rng = random.Random(42)
    for _ in range(200):
        agreed_price = rng.uniform(80, 140)
        reward = reward_engine.compute(
            deal_reached=True,
            agreed_price=agreed_price,
            buyer_max_price=120.0,
            seller_min_price=95.0,
            current_round=rng.randint(1, 10),
            max_rounds=10,
            timed_out=False,
        )
        assert -1.0 <= reward.total <= 1.0, f"Reward out of range: {reward.total}"


def test_reward_engine_never_crashes_on_bad_inputs():
    reward = reward_engine.compute(
        deal_reached=True,
        agreed_price=None,
        buyer_max_price=0,
        seller_min_price=0,
        current_round=0,
        max_rounds=0,
        timed_out=False,
    )
    assert isinstance(reward.total, float)


def test_breakdown_dict_has_all_keys():
    reward = reward_engine.compute(
        deal_reached=True,
        agreed_price=105.0,
        buyer_max_price=120.0,
        seller_min_price=95.0,
        current_round=3,
        max_rounds=10,
        timed_out=False,
    )
    breakdown = reward.to_dict()
    required = [
        "reward/total",
        "reward/savings",
        "reward/efficiency",
        "reward/compliance",
        "metric/deal_reached",
    ]
    for key in required:
        assert key in breakdown


def test_procure_env_uses_reward_engine():
    from agenticpay.openenv_adapter import ProcureEnv

    env = ProcureEnv()
    env.reset()
    _, reward, _, _, _ = env.step("I offer $108. <BUYER_PRICE>108</BUYER_PRICE>")
    assert isinstance(reward, float)
    assert hasattr(env, "_last_reward_breakdown")


def test_extended_env_uses_reward_engine():
    from agenticpay.openenv_adapter.procure_env_extended import ProcureEnvExtended

    env = ProcureEnvExtended()
    env.reset(seed=1)
    _, reward, _, _, _ = env.step("I offer $108. <BUYER_PRICE>108</BUYER_PRICE>")
    assert isinstance(reward, float)
    assert hasattr(env, "_last_reward_breakdown")
