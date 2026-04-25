from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NegotiationObservation:
    # Round information
    current_round: int
    max_rounds: int
    rounds_remaining: int

    # Price information visible to buyer
    seller_last_price: float
    buyer_max_price: float
    initial_seller_price: float

    # Market context
    market_signal_low: float
    market_signal_high: float

    # Conversation history
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Episode state flags
    deal_reached: bool = False
    episode_over: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "rounds_remaining": self.rounds_remaining,
            "seller_last_price": self.seller_last_price,
            "buyer_max_price": self.buyer_max_price,
            "initial_seller_price": self.initial_seller_price,
            "market_signal_low": self.market_signal_low,
            "market_signal_high": self.market_signal_high,
            "conversation_history": self.conversation_history,
            "deal_reached": self.deal_reached,
            "episode_over": self.episode_over,
        }
