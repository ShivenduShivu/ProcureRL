import random
from typing import Any, Dict


class ScriptedSeller:
    """
    Fixed rule-based seller for RL training stability.
    Uses linear concession strategy and never goes below its floor.
    """

    def __init__(
        self,
        seller_min_price: float,
        initial_price: float,
        concession_rate: float = 0.05,
        seed: int = 42,
    ):
        self.seller_min_price = seller_min_price
        self.initial_price = initial_price
        self.concession_rate = concession_rate
        self.current_price = initial_price
        self.rng = random.Random(seed)

    def reset(self):
        self.current_price = self.initial_price

    def respond(self, observation: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        reduction = self.initial_price * self.concession_rate * round_num
        new_price = max(self.seller_min_price, self.initial_price - reduction)

        noise = self.rng.uniform(-0.5, 0.5)
        new_price = max(self.seller_min_price, round(new_price + noise, 2))
        self.current_price = new_price

        buyer_offer = observation.get("last_buyer_price")
        if buyer_offer is not None and buyer_offer >= new_price:
            message = f"I can accept your offer. <SELLER_PRICE>{new_price}</SELLER_PRICE>"
        else:
            message = (
                f"The best I can do is ${new_price:.2f}. "
                f"This already reflects a significant discount. "
                f"<SELLER_PRICE>{new_price}</SELLER_PRICE>"
            )

        return {"message": message, "price": new_price}
