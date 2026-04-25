import random
import re
from typing import Any, Dict, Optional, Tuple

try:
    from gymnasium import spaces
except ImportError:
    spaces = None

from agenticpay.memory.conversation_memory import ConversationMemory

from .action import BuyerAction
from .extended_observation import ExtendedNegotiationObservation
from .extended_seller import ExtendedScriptedSeller
from .procure_env import ProcureEnv
from .scenario_generator import ProcurementScenario, generate_scenario


class ProcureEnvExtended(ProcureEnv):
    """
    Extended procurement environment with:
    - Multi-variable negotiation: price + delivery + quality
    - Competitive pressure signals
    - Procurement policy constraint enforcement
    - Scenario generator for curriculum diversity
    """

    ENV_ID = "ProcureRL-Extended-v0"

    def __init__(self, config: Optional[Dict[str, Any]] = None, difficulty: str = "easy"):
        self.difficulty = difficulty
        self._scenario: Optional[ProcurementScenario] = None
        self._last_competitor_signal = None
        super().__init__(config)
        self.observation_space = self._build_extended_observation_space()

    def _build_extended_observation_space(self):
        if spaces is None:
            base_fields = list(self._build_observation_space()["fields"])
            base_fields.extend(
                [
                    "seller_offered_delivery_days",
                    "buyer_max_delivery_days",
                    "seller_offered_quality_tier",
                    "buyer_min_quality_tier",
                    "competitor_signal",
                    "has_competitor",
                    "policy_budget_ceiling",
                ]
            )
            return {"type": "dict", "fields": base_fields}

        base_price_high = max(
            self.config["initial_seller_price"],
            self.config["buyer_max_price"],
            self.config["true_market_price"] + self.config["market_signal_noise"],
        )
        max_rounds = int(self.config["max_rounds"])
        return spaces.Dict(
            {
                "current_round": spaces.Discrete(max_rounds + 1),
                "max_rounds": spaces.Discrete(max_rounds + 1),
                "rounds_remaining": spaces.Discrete(max_rounds + 1),
                "seller_last_price": spaces.Box(low=0.0, high=base_price_high * 2, shape=(), dtype=float),
                "buyer_max_price": spaces.Box(low=0.0, high=base_price_high * 2, shape=(), dtype=float),
                "initial_seller_price": spaces.Box(low=0.0, high=base_price_high * 2, shape=(), dtype=float),
                "market_signal_low": spaces.Box(low=0.0, high=base_price_high * 2, shape=(), dtype=float),
                "market_signal_high": spaces.Box(low=0.0, high=base_price_high * 2, shape=(), dtype=float),
                "conversation_history": spaces.Sequence(
                    spaces.Dict(
                        {
                            "role": spaces.Text(max_length=32),
                            "content": spaces.Text(max_length=4096),
                            "round": spaces.Discrete(max_rounds + 1),
                            "metadata": spaces.Dict({}),
                        }
                    )
                ),
                "deal_reached": spaces.Discrete(2),
                "episode_over": spaces.Discrete(2),
                "seller_offered_delivery_days": spaces.Discrete(366),
                "buyer_max_delivery_days": spaces.Discrete(366),
                "seller_offered_quality_tier": spaces.Discrete(4),
                "buyer_min_quality_tier": spaces.Discrete(4),
                "competitor_signal": spaces.Box(low=0.0, high=base_price_high * 2, shape=(), dtype=float),
                "has_competitor": spaces.Discrete(2),
                "policy_budget_ceiling": spaces.Box(low=0.0, high=base_price_high * 2, shape=(), dtype=float),
            }
        )

    def _generate_competitor_signal(self) -> Optional[float]:
        if not self._scenario or not self._scenario.has_competitor:
            self._last_competitor_signal = None
            return None

        base = self._scenario.true_market_price + self._scenario.competitor_price_offset
        self._last_competitor_signal = round(base + self.rng.uniform(-2, 2), 2)
        return self._last_competitor_signal

    def _build_extended_observation(self, seller_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        market_low, market_high = self._last_market_signal
        seller_delivery = self._scenario.initial_delivery_days if self._scenario else 30
        seller_quality = 2
        buyer_max_delivery = self._scenario.buyer_max_delivery_days if self._scenario else 21
        buyer_min_quality = self._scenario.buyer_min_quality_tier if self._scenario else 2

        if seller_response:
            seller_delivery = seller_response.get("delivery_days", seller_delivery)
            seller_quality = seller_response.get("quality_tier", seller_quality)

        obs = ExtendedNegotiationObservation(
            current_round=self.current_round,
            max_rounds=self.config["max_rounds"],
            rounds_remaining=self.config["max_rounds"] - self.current_round,
            seller_last_price=float(self.last_seller_price),
            buyer_max_price=float(self.config["buyer_max_price"]),
            initial_seller_price=float(self.config["initial_seller_price"]),
            market_signal_low=float(market_low),
            market_signal_high=float(market_high),
            conversation_history=self.memory.get_history(),
            deal_reached=self.terminated,
            episode_over=self.terminated or self.truncated,
            seller_offered_delivery_days=int(seller_delivery),
            buyer_max_delivery_days=int(buyer_max_delivery),
            seller_offered_quality_tier=int(seller_quality),
            buyer_min_quality_tier=int(buyer_min_quality),
            competitor_signal=self._last_competitor_signal,
            has_competitor=bool(self._scenario.has_competitor if self._scenario else False),
            policy_budget_ceiling=float(self.config["buyer_max_price"]),
        )
        return obs.to_dict()

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)

        scenario_seed = self.rng.randint(0, 999999)
        self._scenario = generate_scenario(self.difficulty, seed=scenario_seed)

        self.config.update(
            {
                "initial_seller_price": self._scenario.initial_seller_price,
                "seller_min_price": self._scenario.seller_min_price,
                "buyer_max_price": self._scenario.buyer_max_price,
                "true_market_price": self._scenario.true_market_price,
                "max_rounds": self._scenario.max_rounds,
                "price_tolerance": self._scenario.price_tolerance,
            }
        )

        self.seller = ExtendedScriptedSeller(
            seller_min_price=self._scenario.seller_min_price,
            initial_price=self._scenario.initial_seller_price,
            initial_delivery_days=self._scenario.initial_delivery_days,
            min_delivery_days=14,
            initial_quality_tier=2,
            concession_rate=self.config["concession_rate"],
            seed=scenario_seed,
        )

        self.observation_space = self._build_extended_observation_space()
        self._reset_state()
        self.seller.reset()
        self.memory = ConversationMemory()
        self._generate_market_signal()
        self._generate_competitor_signal()

        obs = self._build_extended_observation()
        info = {
            "status": "running",
            "product": self._scenario.product_name,
            "difficulty": self.difficulty,
            "scenario": {
                "product_name": self._scenario.product_name,
                "product_category": self._scenario.product_category,
                "difficulty": self._scenario.difficulty,
            },
        }
        return obs, info

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is over. Call reset() first.")

        buyer_action = BuyerAction.from_text(action)
        self.last_buyer_price = buyer_action.offered_price
        buyer_delivery = self._extract_int_tag(action, "BUYER_DELIVERY")
        buyer_quality = self._extract_int_tag(action, "BUYER_QUALITY")

        self.memory.add_message("buyer", action, self.current_round)

        seller_obs = {
            "last_buyer_price": self.last_buyer_price,
            "buyer_message": action,
        }
        seller_response = self.seller.respond(seller_obs, self.current_round)
        self.last_seller_price = float(seller_response["price"])
        self.seller_last_response = seller_response["message"]
        self.memory.add_message("seller", self.seller_last_response, self.current_round)

        deal_reached = self._check_agreement(self.last_buyer_price, self.last_seller_price)

        self.current_round += 1

        if deal_reached:
            self.agreed_price = self._compute_agreed_price(self.last_buyer_price, self.last_seller_price)
            self.terminated = True
        elif self.current_round >= self.config["max_rounds"]:
            self.truncated = True

        reward = float(
            self._compute_extended_reward(
                deal_reached,
                self.last_buyer_price,
                buyer_delivery,
                buyer_quality,
                seller_response,
            )
        )

        self._generate_market_signal()
        self._generate_competitor_signal()
        obs = self._build_extended_observation(seller_response=seller_response)
        info = {
            "status": "deal" if self.terminated else "timeout" if self.truncated else "running",
            "agreed_price": self.agreed_price,
            "delivery_days": seller_response.get("delivery_days"),
            "quality_tier": seller_response.get("quality_tier"),
            "deal_reached": deal_reached,
            "budget_ceiling": self.config["buyer_max_price"],
        }
        return obs, reward, self.terminated, self.truncated, info

    def _extract_int_tag(self, text: str, tag: str) -> Optional[int]:
        match = re.search(rf"<{tag}>\s*(\d+)\s*</{tag}>", text)
        return int(match.group(1)) if match else None

    def _compute_extended_reward(
        self,
        deal_reached: bool,
        buyer_price: Optional[float],
        buyer_delivery: Optional[int],
        buyer_quality: Optional[int],
        seller_response: Dict[str, Any],
    ) -> float:
        if not deal_reached:
            return -0.5 if self.truncated else 0.0

        if self.agreed_price is None:
            return 0.0

        max_price = float(self.config["buyer_max_price"])
        if self.agreed_price > max_price:
            return -1.0

        savings = (max_price - self.agreed_price) / max_price

        if buyer_delivery is not None and buyer_delivery < seller_response.get("delivery_days", buyer_delivery):
            savings -= 0.02
        if buyer_quality is not None and buyer_quality > seller_response.get("quality_tier", buyer_quality):
            savings -= 0.02

        return round(float(savings), 4)

    def state(self) -> Dict[str, Any]:
        base_state = super().state()
        base_state.update(
            {
                "difficulty": self.difficulty,
                "competitor_signal": self._last_competitor_signal,
                "scenario": self._scenario.__dict__ if self._scenario else None,
            }
        )
        return base_state
