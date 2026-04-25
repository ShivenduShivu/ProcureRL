from dataclasses import dataclass
import re
from typing import Optional


@dataclass
class BuyerAction:
    message: str
    offered_price: Optional[float] = None

    @classmethod
    def from_text(cls, text: str) -> "BuyerAction":
        """Parse raw LLM output into a BuyerAction."""
        tag_match = re.search(r"<BUYER_PRICE>\s*([\d.]+)\s*</BUYER_PRICE>", text)
        if tag_match:
            price = float(tag_match.group(1))
            return cls(message=text, offered_price=price)

        dollar_match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", text)
        if dollar_match:
            price = float(dollar_match.group(1).replace(",", ""))
            return cls(message=text, offered_price=price)

        return cls(message=text, offered_price=None)
