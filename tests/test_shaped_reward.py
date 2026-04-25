from agenticpay.envs.reward_engine import reward_engine


def test_no_price_worse_than_good_offer():
    r_none = reward_engine.compute_shaped(
        deal_reached=False,
        agreed_price=None,
        offered_price=None,
        buyer_max_price=120,
        seller_current_price=125,
        seller_min_price=95,
        current_round=1,
        max_rounds=10,
        timed_out=False,
    )
    r_good = reward_engine.compute_shaped(
        deal_reached=False,
        agreed_price=None,
        offered_price=105,
        buyer_max_price=120,
        seller_current_price=125,
        seller_min_price=95,
        current_round=1,
        max_rounds=10,
        timed_out=False,
    )
    assert r_good.total > r_none.total


def test_over_budget_gives_negative_one():
    r = reward_engine.compute_shaped(
        deal_reached=False,
        agreed_price=None,
        offered_price=130,
        buyer_max_price=120,
        seller_current_price=125,
        seller_min_price=95,
        current_round=1,
        max_rounds=10,
        timed_out=False,
    )
    assert r.total == -1.0


def test_aggressive_offer_better_than_high_offer():
    r_low = reward_engine.compute_shaped(
        deal_reached=False,
        agreed_price=None,
        offered_price=100,
        buyer_max_price=120,
        seller_current_price=125,
        seller_min_price=95,
        current_round=1,
        max_rounds=10,
        timed_out=False,
    )
    r_high = reward_engine.compute_shaped(
        deal_reached=False,
        agreed_price=None,
        offered_price=118,
        buyer_max_price=120,
        seller_current_price=125,
        seller_min_price=95,
        current_round=1,
        max_rounds=10,
        timed_out=False,
    )
    assert r_low.total > r_high.total


def test_deal_uses_full_reward():
    r = reward_engine.compute_shaped(
        deal_reached=True,
        agreed_price=105,
        offered_price=105,
        buyer_max_price=120,
        seller_current_price=105,
        seller_min_price=95,
        current_round=3,
        max_rounds=10,
        timed_out=False,
    )
    assert r.deal_reached is True
    assert r.total > 0
