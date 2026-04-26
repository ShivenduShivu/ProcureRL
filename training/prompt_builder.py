from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """You are a professional procurement officer negotiating a purchase.
Your goal is to secure the best possible deal within your procurement budget.

RULES:
1. You MUST include your price offer in the format:
   <BUYER_PRICE>NUMBER</BUYER_PRICE>
2. You MUST NOT agree to any price above your budget ceiling.
3. Be strategic - anchor low, make concessions slowly,
   use market data if available.
4. If a competitor signal is available, mention it to create
   competitive pressure.
5. Aim to close the deal in as few rounds as possible.

OUTPUT FORMAT:
Write your negotiation message naturally, then end with
the price tag.
Example: 'Based on market data, I can offer $105 for
immediate payment. <BUYER_PRICE>105</BUYER_PRICE>'
"""


def build_prompt(
    observation: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the full prompt string for the LLM buyer agent."""
    obs = observation
    round_num = obs.get("current_round", 0)
    max_rounds = obs.get("max_rounds", 10)
    seller_price = obs.get("seller_last_price", 0)
    budget = obs.get("buyer_max_price", 0)
    market_low = obs.get("market_signal_low", 0)
    market_high = obs.get("market_signal_high", 0)
    competitor = obs.get("competitor_signal")
    history = conversation_history if conversation_history is not None else obs.get("conversation_history", [])
    seller_min_approx = obs.get("seller_last_price", market_low) * 0.75
    target_open = seller_min_approx + (budget - seller_min_approx) * 0.20

    context_lines = [
        f"Round {round_num + 1} of {max_rounds}.",
        f"Seller current price: ${seller_price:.2f}.",
        f"Your budget ceiling: ${budget:.2f} (DO NOT exceed this).",
        f"Market price range: ${market_low:.2f} - ${market_high:.2f}.",
        f"Your target opening offer: around ${target_open:.2f} "
        f"(strategically positioned below seller price and market)",
    ]
    if competitor is not None:
        context_lines.append(f"Competing supplier quote: ${competitor:.2f}.")

    delivery = obs.get("seller_offered_delivery_days")
    quality = obs.get("seller_offered_quality_tier")
    max_delivery = obs.get("buyer_max_delivery_days")
    min_quality = obs.get("buyer_min_quality_tier")
    if delivery is not None:
        context_lines.append(
            f"Seller delivery offer: {delivery} days (your max: {max_delivery} days)."
        )
    if quality is not None:
        context_lines.append(
            f"Seller quality tier: {quality}/3 (your minimum: {min_quality}/3)."
        )

    context = "\n".join(context_lines)

    history_text = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        history_text = "\nConversation so far:\n" + "\n".join(lines)

    user_message = (
        f"Current situation:\n{context}{history_text}\n\n"
        "Your move. Write your negotiation response:"
    )
    return user_message


def format_for_trl(observation: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format observation as a chat message list for GRPOTrainer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(observation)},
    ]
