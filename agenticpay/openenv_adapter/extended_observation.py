from dataclasses import dataclass
from typing import Any, Dict, Optional

from .observation import NegotiationObservation


@dataclass
class ExtendedNegotiationObservation(NegotiationObservation):
    # Delivery dimension
    seller_offered_delivery_days: int = 30
    buyer_max_delivery_days: int = 21

    # Quality dimension
    seller_offered_quality_tier: int = 2
    buyer_min_quality_tier: int = 2

    # Competitive pressure
    competitor_signal: Optional[float] = None
    has_competitor: bool = False

    # Compliance
    policy_budget_ceiling: float = 120.0
    last_buyer_price: Optional[float] = None
    seller_concession_amount: float = 0.0
    normalized_budget_gap: float = 0.0
    rounds_remaining_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "seller_offered_delivery_days": self.seller_offered_delivery_days,
                "buyer_max_delivery_days": self.buyer_max_delivery_days,
                "seller_offered_quality_tier": self.seller_offered_quality_tier,
                "buyer_min_quality_tier": self.buyer_min_quality_tier,
                "competitor_signal": self.competitor_signal,
                "has_competitor": self.has_competitor,
                "policy_budget_ceiling": self.policy_budget_ceiling,
                "last_buyer_price": self.last_buyer_price,
                "seller_concession_amount": self.seller_concession_amount,
                "normalized_budget_gap": self.normalized_budget_gap,
                "rounds_remaining_ratio": self.rounds_remaining_ratio,
            }
        )
        return base
