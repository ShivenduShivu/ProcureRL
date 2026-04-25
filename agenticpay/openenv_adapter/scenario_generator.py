import random
from dataclasses import dataclass


@dataclass
class ProcurementScenario:
    product_name: str
    product_category: str
    initial_seller_price: float
    seller_min_price: float
    buyer_max_price: float
    true_market_price: float
    initial_delivery_days: int
    buyer_max_delivery_days: int
    buyer_min_quality_tier: int
    has_competitor: bool
    competitor_price_offset: float
    difficulty: str
    max_rounds: int
    price_tolerance: float


SCENARIO_TEMPLATES = [
    {
        "product_name": "Industrial Steel Sheets",
        "product_category": "raw_materials",
        "base_market": 100.0,
        "seller_markup": 0.30,
        "buyer_budget_ratio": 0.20,
    },
    {
        "product_name": "Cloud Computing Credits (Annual)",
        "product_category": "services",
        "base_market": 5000.0,
        "seller_markup": 0.25,
        "buyer_budget_ratio": 0.15,
    },
    {
        "product_name": "Office Furniture Package",
        "product_category": "assets",
        "base_market": 3000.0,
        "seller_markup": 0.35,
        "buyer_budget_ratio": 0.18,
    },
    {
        "product_name": "API Data Feed Subscription",
        "product_category": "services",
        "base_market": 800.0,
        "seller_markup": 0.40,
        "buyer_budget_ratio": 0.20,
    },
    {
        "product_name": "Safety Equipment Bulk Order",
        "product_category": "equipment",
        "base_market": 2000.0,
        "seller_markup": 0.28,
        "buyer_budget_ratio": 0.16,
    },
]

DIFFICULTY_CONFIGS = {
    "easy": {"max_rounds": 10, "price_tolerance": 2.0, "has_competitor_prob": 0.3},
    "medium": {"max_rounds": 8, "price_tolerance": 1.0, "has_competitor_prob": 0.6},
    "hard": {"max_rounds": 6, "price_tolerance": 0.5, "has_competitor_prob": 0.8},
}


def generate_scenario(difficulty: str = "easy", seed: int = None) -> ProcurementScenario:
    if difficulty not in DIFFICULTY_CONFIGS:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    rng = random.Random(seed)
    template = rng.choice(SCENARIO_TEMPLATES)
    diff_cfg = DIFFICULTY_CONFIGS[difficulty]

    market = template["base_market"]
    markup = template["seller_markup"]
    budget_ratio = template["buyer_budget_ratio"]

    return ProcurementScenario(
        product_name=template["product_name"],
        product_category=template["product_category"],
        initial_seller_price=round(market * (1 + markup), 2),
        seller_min_price=round(market * 0.90, 2),
        buyer_max_price=round(market * (1 + budget_ratio), 2),
        true_market_price=market,
        initial_delivery_days=rng.randint(21, 45),
        buyer_max_delivery_days=rng.randint(14, 30),
        buyer_min_quality_tier=rng.choice([1, 2]),
        has_competitor=rng.random() < diff_cfg["has_competitor_prob"],
        competitor_price_offset=round(rng.uniform(-5, 10), 2),
        difficulty=difficulty,
        max_rounds=diff_cfg["max_rounds"],
        price_tolerance=diff_cfg["price_tolerance"],
    )
