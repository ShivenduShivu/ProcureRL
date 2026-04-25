import random
from typing import Any, Dict, Optional, Tuple

try:
    from gymnasium import spaces
except ImportError:
    spaces = None

try:
    from openenv import Environment
except ImportError:
    class Environment:
        pass

from agenticpay.envs.reward_engine import reward_engine
from agenticpay.memory.conversation_memory import ConversationMemory

from .action import BuyerAction
from .observation import NegotiationObservation
from .scripted_seller import ScriptedSeller


class ProcureEnv(Environment):
    """
    OpenEnv-compatible procurement negotiation environment.
    Trains one buyer policy against a fixed scripted seller.
    """

    ENV_ID = "ProcureRL-v0"

    DEFAULT_CONFIG = {
        "initial_seller_price": 130.0,
        "seller_min_price": 95.0,
        "buyer_max_price": 120.0,
        "max_rounds": 10,
        "price_tolerance": 1.0,
        "market_signal_noise": 5.0,
        "true_market_price": 105.0,
        "concession_rate": 0.05,
        "seed": 42,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.seller = ScriptedSeller(
            seller_min_price=self.config["seller_min_price"],
            initial_price=self.config["initial_seller_price"],
            concession_rate=self.config["concession_rate"],
            seed=self.config["seed"],
        )
        self.memory = ConversationMemory()
        self.rng = random.Random(self.config["seed"])
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()
        self._last_market_signal = (
            self.config["true_market_price"],
            self.config["true_market_price"],
        )
        self._reset_state()

    def _build_observation_space(self):
        if spaces is None:
            return {
                "type": "dict",
                "fields": [
                    "current_round",
                    "max_rounds",
                    "rounds_remaining",
                    "seller_last_price",
                    "buyer_max_price",
                    "initial_seller_price",
                    "market_signal_low",
                    "market_signal_high",
                    "conversation_history",
                    "deal_reached",
                    "episode_over",
                ],
            }

        max_rounds = int(self.config["max_rounds"])
        max_price = max(
            self.config["initial_seller_price"],
            self.config["buyer_max_price"],
            self.config["true_market_price"] + self.config["market_signal_noise"],
        )
        return spaces.Dict(
            {
                "current_round": spaces.Discrete(max_rounds + 1),
                "max_rounds": spaces.Discrete(max_rounds + 1),
                "rounds_remaining": spaces.Discrete(max_rounds + 1),
                "seller_last_price": spaces.Box(low=0.0, high=max_price * 2, shape=(), dtype=float),
                "buyer_max_price": spaces.Box(low=0.0, high=max_price * 2, shape=(), dtype=float),
                "initial_seller_price": spaces.Box(low=0.0, high=max_price * 2, shape=(), dtype=float),
                "market_signal_low": spaces.Box(low=0.0, high=max_price * 2, shape=(), dtype=float),
                "market_signal_high": spaces.Box(low=0.0, high=max_price * 2, shape=(), dtype=float),
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
            }
        )

    def _build_action_space(self):
        if spaces is None:
            return {"type": "text", "description": "Buyer negotiation message"}
        return spaces.Text(max_length=4096)

    def _reset_state(self):
        self.current_round = 0
        self.terminated = False
        self.truncated = False
        self.agreed_price = None
        self.last_buyer_price = None
        self.last_seller_price = float(self.config["initial_seller_price"])
        self.seller_last_response = None

    def _generate_market_signal(self) -> Tuple[float, float]:
        noise = self.config["market_signal_noise"]
        true_price = self.config["true_market_price"]
        low = true_price - self.rng.uniform(0, noise)
        high = true_price + self.rng.uniform(0, noise)
        self._last_market_signal = (round(low, 2), round(high, 2))
        return self._last_market_signal

    def _build_observation(self) -> Dict[str, Any]:
        market_low, market_high = self._last_market_signal
        obs = NegotiationObservation(
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
        )
        return obs.to_dict()

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
            self.seller = ScriptedSeller(
                seller_min_price=self.config["seller_min_price"],
                initial_price=self.config["initial_seller_price"],
                concession_rate=self.config["concession_rate"],
                seed=seed,
            )

        self._reset_state()
        self.seller.reset()
        self.memory = ConversationMemory()
        self._generate_market_signal()

        obs = self._build_observation()
        info = {"status": "running", "round": 0}
        return obs, info

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called on finished episode. Call reset() first.")

        buyer_action = BuyerAction.from_text(action)
        self.last_buyer_price = buyer_action.offered_price

        self.memory.add_message(
            role="buyer",
            content=buyer_action.message,
            round=self.current_round,
        )

        seller_obs = {"last_buyer_price": self.last_buyer_price}
        seller_response = self.seller.respond(seller_obs, self.current_round)
        self.last_seller_price = float(seller_response["price"])
        self.seller_last_response = seller_response["message"]

        self.memory.add_message(
            role="seller",
            content=self.seller_last_response,
            round=self.current_round,
        )

        deal_reached = self._check_agreement(self.last_buyer_price, self.last_seller_price)

        self.current_round += 1

        if deal_reached:
            self.agreed_price = self._compute_agreed_price(self.last_buyer_price, self.last_seller_price)
            self.terminated = True
        elif self.current_round >= self.config["max_rounds"]:
            self.truncated = True

        reward = float(self._compute_reward(deal_reached, self.last_buyer_price))

        self._generate_market_signal()
        obs = self._build_observation()
        info = {
            "status": "deal" if self.terminated else "timeout" if self.truncated else "running",
            "round": self.current_round,
            "agreed_price": self.agreed_price,
            "buyer_max_price": self.config["buyer_max_price"],
            "seller_min_price": self.config["seller_min_price"],
            "deal_reached": deal_reached,
            "seller_message": self.seller_last_response,
        }
        return obs, reward, self.terminated, self.truncated, info

    def _check_agreement(self, buyer_price: Optional[float], seller_price: float) -> bool:
        if buyer_price is None:
            return False
        tolerance = self.config["price_tolerance"]
        return abs(buyer_price - seller_price) <= tolerance or buyer_price >= seller_price

    def _compute_agreed_price(self, buyer_price: Optional[float], seller_price: float) -> float:
        if buyer_price is None:
            return float(seller_price)
        if buyer_price >= seller_price:
            return float(seller_price)
        return float((buyer_price + seller_price) / 2.0)

    def _compute_reward(self, deal_reached: bool, buyer_price: Optional[float]) -> float:
        breakdown = reward_engine.compute(
            deal_reached=deal_reached,
            agreed_price=self.agreed_price,
            buyer_max_price=self.config["buyer_max_price"],
            seller_min_price=self.config["seller_min_price"],
            current_round=self.current_round,
            max_rounds=self.config["max_rounds"],
            timed_out=self.truncated,
        )
        self._last_reward_breakdown = breakdown
        return breakdown.total

    def state(self) -> Dict[str, Any]:
        market_low, market_high = self._last_market_signal
        return {
            "current_round": self.current_round,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "agreed_price": self.agreed_price,
            "last_buyer_price": self.last_buyer_price,
            "last_seller_price": self.last_seller_price,
            "seller_last_response": self.seller_last_response,
            "market_signal_low": market_low,
            "market_signal_high": market_high,
            "config": self.config,
        }

    def close(self):
        return None
