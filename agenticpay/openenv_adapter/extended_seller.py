from typing import Any, Dict

from .scripted_seller import ScriptedSeller


class ExtendedScriptedSeller(ScriptedSeller):
    """
    Multi-variable scripted seller.
    Negotiates price, delivery days, and quality tier.
    """

    def __init__(
        self,
        seller_min_price: float,
        initial_price: float,
        initial_delivery_days: int = 30,
        min_delivery_days: int = 14,
        initial_quality_tier: int = 2,
        concession_rate: float = 0.05,
        seed: int = 42,
    ):
        super().__init__(seller_min_price, initial_price, concession_rate, seed)
        self.initial_delivery_days = initial_delivery_days
        self.min_delivery_days = min_delivery_days
        self.initial_quality_tier = initial_quality_tier
        self.current_delivery_days = initial_delivery_days
        self.current_quality_tier = initial_quality_tier

    def reset(self):
        super().reset()
        self.current_delivery_days = self.initial_delivery_days
        self.current_quality_tier = self.initial_quality_tier

    def respond(self, observation: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        base_response = super().respond(observation, round_num)

        delivery_improvement = min(
            round_num,
            self.initial_delivery_days - self.min_delivery_days,
        )
        self.current_delivery_days = max(
            self.min_delivery_days,
            self.initial_delivery_days - delivery_improvement,
        )

        buyer_message = observation.get("buyer_message", "").lower()
        if "premium" in buyer_message or "quality" in buyer_message:
            self.current_quality_tier = min(3, self.current_quality_tier + 1)

        price = base_response["price"]
        message = (
            f"My offer: ${price:.2f}, delivery in {self.current_delivery_days} days, "
            f"quality tier {self.current_quality_tier}/3. "
            f"<SELLER_PRICE>{price}</SELLER_PRICE>"
            f"<SELLER_DELIVERY>{self.current_delivery_days}</SELLER_DELIVERY>"
            f"<SELLER_QUALITY>{self.current_quality_tier}</SELLER_QUALITY>"
        )
        return {
            "message": message,
            "price": price,
            "delivery_days": self.current_delivery_days,
            "quality_tier": self.current_quality_tier,
        }
