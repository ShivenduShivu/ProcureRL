from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for logging and debugging."""

    total: float
    r_savings: float
    r_efficiency: float
    r_compliance: float
    r_deal_quality: float
    penalty_constraint: float
    penalty_timeout: float
    deal_reached: bool
    constraint_violated: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward/total": self.total,
            "reward/savings": self.r_savings,
            "reward/efficiency": self.r_efficiency,
            "reward/compliance": self.r_compliance,
            "reward/deal_quality": self.r_deal_quality,
            "reward/penalty_constraint": self.penalty_constraint,
            "reward/penalty_timeout": self.penalty_timeout,
            "metric/deal_reached": float(self.deal_reached),
            "metric/constraint_violated": float(self.constraint_violated),
        }


class ProcureRewardEngine:
    """
    Multi-component reward engine for ProcureRL.
    All components return values in [-1.0, 1.0].
    Hard constraints override other rewards.
    """

    W_SAVINGS = 0.40
    W_EFFICIENCY = 0.20
    W_COMPLIANCE = 0.20
    W_DEAL_QUALITY = 0.20

    PENALTY_CONSTRAINT_VIOLATION = -1.0
    PENALTY_TIMEOUT = -0.3
    PENALTY_NO_PRICE_OFFER = -0.1

    def compute(
        self,
        deal_reached: bool,
        agreed_price: Optional[float],
        buyer_max_price: float,
        seller_min_price: float,
        current_round: int,
        max_rounds: int,
        timed_out: bool,
        agreed_delivery_days: Optional[int] = None,
        buyer_max_delivery_days: Optional[int] = None,
        agreed_quality_tier: Optional[int] = None,
        buyer_min_quality_tier: Optional[int] = None,
    ) -> RewardBreakdown:
        try:
            return self._compute_safe(
                deal_reached=deal_reached,
                agreed_price=agreed_price,
                buyer_max_price=buyer_max_price,
                seller_min_price=seller_min_price,
                current_round=current_round,
                max_rounds=max_rounds,
                timed_out=timed_out,
                agreed_delivery_days=agreed_delivery_days,
                buyer_max_delivery_days=buyer_max_delivery_days,
                agreed_quality_tier=agreed_quality_tier,
                buyer_min_quality_tier=buyer_min_quality_tier,
            )
        except Exception:
            return RewardBreakdown(
                total=0.0,
                r_savings=0.0,
                r_efficiency=0.0,
                r_compliance=0.0,
                r_deal_quality=0.0,
                penalty_constraint=0.0,
                penalty_timeout=0.0,
                deal_reached=False,
                constraint_violated=False,
            )

    def _compute_safe(
        self,
        deal_reached,
        agreed_price,
        buyer_max_price,
        seller_min_price,
        current_round,
        max_rounds,
        timed_out,
        agreed_delivery_days,
        buyer_max_delivery_days,
        agreed_quality_tier,
        buyer_min_quality_tier,
    ) -> RewardBreakdown:
        if timed_out or not deal_reached:
            return RewardBreakdown(
                total=self.PENALTY_TIMEOUT,
                r_savings=0.0,
                r_efficiency=0.0,
                r_compliance=0.0,
                r_deal_quality=0.0,
                penalty_constraint=0.0,
                penalty_timeout=self.PENALTY_TIMEOUT,
                deal_reached=False,
                constraint_violated=False,
            )

        constraint_violated = False
        penalty_constraint = 0.0
        if agreed_price is not None and agreed_price > buyer_max_price:
            constraint_violated = True
            penalty_constraint = self.PENALTY_CONSTRAINT_VIOLATION
            return RewardBreakdown(
                total=penalty_constraint,
                r_savings=0.0,
                r_efficiency=0.0,
                r_compliance=0.0,
                r_deal_quality=0.0,
                penalty_constraint=penalty_constraint,
                penalty_timeout=0.0,
                deal_reached=True,
                constraint_violated=True,
            )

        deal_range = buyer_max_price - seller_min_price
        if deal_range > 0 and agreed_price is not None:
            savings = (buyer_max_price - agreed_price) / deal_range
            r_savings = max(0.0, min(1.0, savings))
        else:
            r_savings = 0.0

        if max_rounds > 0:
            r_efficiency = 1.0 - (current_round / max_rounds)
        else:
            r_efficiency = 0.0
        r_efficiency = max(0.0, min(1.0, r_efficiency))

        r_compliance = 1.0 if not constraint_violated else 0.0

        r_deal_quality = 0.0
        quality_components = []
        if agreed_delivery_days is not None and buyer_max_delivery_days is not None:
            if buyer_max_delivery_days > 0 and agreed_delivery_days <= buyer_max_delivery_days:
                delivery_score = 1.0 - (agreed_delivery_days / buyer_max_delivery_days)
                quality_components.append(max(0.0, min(1.0, delivery_score)))
            else:
                quality_components.append(0.0)

        if agreed_quality_tier is not None and buyer_min_quality_tier is not None:
            if agreed_quality_tier >= buyer_min_quality_tier:
                quality_components.append(1.0)
            else:
                quality_components.append(0.0)

        r_deal_quality = (
            sum(quality_components) / len(quality_components)
            if quality_components
            else 0.5
        )

        total = (
            self.W_SAVINGS * r_savings
            + self.W_EFFICIENCY * r_efficiency
            + self.W_COMPLIANCE * r_compliance
            + self.W_DEAL_QUALITY * r_deal_quality
        )
        total = round(max(-1.0, min(1.0, total)), 4)

        return RewardBreakdown(
            total=total,
            r_savings=round(r_savings, 4),
            r_efficiency=round(r_efficiency, 4),
            r_compliance=round(r_compliance, 4),
            r_deal_quality=round(r_deal_quality, 4),
            penalty_constraint=penalty_constraint,
            penalty_timeout=0.0,
            deal_reached=True,
            constraint_violated=constraint_violated,
        )


reward_engine = ProcureRewardEngine()
