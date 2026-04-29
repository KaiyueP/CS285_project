"""
REINFORCE (Monte-Carlo policy gradient) for selecting a DFA for a reaction.

Reward = -absolute_error vs reference for the sampled functional (from GSCDB
Reaction_Energies during training). At inference, policy probabilities rank functionals.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _xavier(n_in: int, n_out: int) -> np.ndarray:
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))


class ReinforcePolicyAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_layers: Tuple[int, ...] = (128, 64),
        learning_rate: float = 0.002,
        entropy_coef: float = 0.0,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.layer_sizes = list(hidden_layers) + [n_actions]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        prev = state_dim
        for h in self.layer_sizes:
            self.weights.append(_xavier(prev, h))
            self.biases.append(np.zeros((1, h)))
            prev = h

    def _forward(
        self,
        x: np.ndarray,
        return_cache: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """x: (batch, state_dim) -> logits (batch, n_actions)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        activations = [x]
        pre = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            pre.append(z)
            if i < len(self.weights) - 1:
                a = np.maximum(0.0, z)
            else:
                a = z
            activations.append(a)
        logits = activations[-1]
        if return_cache:
            return logits, activations, pre
        return (logits,)

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        logits = self._forward(state)[0]
        return _softmax(logits)[0]

    def sample_action(self, state: np.ndarray) -> Tuple[int, float, np.ndarray]:
        probs = self.action_probs(state)
        a = int(np.random.choice(self.n_actions, p=probs))
        log_pi = float(np.log(probs[a] + 1e-12))
        return a, log_pi, probs

    def greedy_action(self, state: np.ndarray) -> int:
        return int(np.argmax(self.action_probs(state)))

    def reinforce_step(
        self,
        state: np.ndarray,
        action: int,
        advantage: float,
    ) -> float:
        """
        One REINFORCE update: minimize -(advantage * log pi(a|s) + entropy_coef * H(pi)).
        Returns approximate loss scalar.
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        logits, activations, pre = self._forward(state, return_cache=True)
        probs = _softmax(logits)
        log_probs = np.log(probs + 1e-12)

        one_hot = np.zeros_like(probs)
        one_hot[0, action] = 1.0
        # d/dz of (-adv * log pi_a - c * H); H = -sum p log p, dH/dz_i = p_i (log p_i + H)
        H = float(-(probs * log_probs).sum())
        if self.entropy_coef != 0.0:
            dH_dlogits = probs * (log_probs + H)
            d_logits = advantage * (probs - one_hot) - self.entropy_coef * dH_dlogits
        else:
            d_logits = advantage * (probs - one_hot)

        loss = float(
            -advantage * log_probs[0, action] - self.entropy_coef * H
        )

        batch = state.shape[0]
        delta = d_logits / max(1, batch)

        grad_w: List[Optional[np.ndarray]] = [None] * len(self.weights)
        grad_b: List[Optional[np.ndarray]] = [None] * len(self.biases)

        for layer_idx in reversed(range(len(self.weights))):
            a_prev = activations[layer_idx]
            grad_w[layer_idx] = a_prev.T @ delta
            grad_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)
            if layer_idx > 0:
                da_prev = delta @ self.weights[layer_idx].T
                relu_mask = (pre[layer_idx - 1] > 0).astype(np.float64)
                delta = da_prev * relu_mask

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

        return loss

    def reinforce_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> float:
        """
        Batched REINFORCE with mean-centered advantages (states: B×d, actions: B, adv: B).
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        bsz = states.shape[0]
        logits, activations, pre = self._forward(states, return_cache=True)
        probs = _softmax(logits)
        log_probs = np.log(probs + 1e-12)

        adv = advantages.reshape(bsz, 1).astype(np.float64)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(bsz), actions] = 1.0

        H_row = -(probs * log_probs).sum(axis=1, keepdims=True)
        if self.entropy_coef != 0.0:
            dH = probs * (log_probs + H_row)
            d_logits = adv * (probs - one_hot) - self.entropy_coef * dH
        else:
            d_logits = adv * (probs - one_hot)

        loss = float(
            -(adv.squeeze() * log_probs[np.arange(bsz), actions]).mean()
            - self.entropy_coef * H_row.mean()
        )

        delta = d_logits / max(1, bsz)

        grad_w: List[Optional[np.ndarray]] = [None] * len(self.weights)
        grad_b: List[Optional[np.ndarray]] = [None] * len(self.biases)

        for layer_idx in reversed(range(len(self.weights))):
            a_prev = activations[layer_idx]
            grad_w[layer_idx] = a_prev.T @ delta
            grad_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)
            if layer_idx > 0:
                da_prev = delta @ self.weights[layer_idx].T
                relu_mask = (pre[layer_idx - 1] > 0).astype(np.float64)
                delta = da_prev * relu_mask

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

        return loss

    def supervised_cross_entropy_batch(
        self,
        states: np.ndarray,
        target_actions: np.ndarray,
    ) -> float:
        """Multi-class cross-entropy toward target functional indices (warm-start)."""
        if states.ndim == 1:
            states = states.reshape(1, -1)
        bsz = states.shape[0]
        logits, activations, pre = self._forward(states, return_cache=True)
        probs = _softmax(logits)
        log_probs = np.log(probs + 1e-12)

        t = np.asarray(target_actions, dtype=np.int64).reshape(bsz)
        selected = log_probs[np.arange(bsz), t]
        loss = float(-selected.mean())

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(bsz), t] = 1.0
        d_logits = (probs - one_hot) / max(1, bsz)

        delta = d_logits
        grad_w: List[Optional[np.ndarray]] = [None] * len(self.weights)
        grad_b: List[Optional[np.ndarray]] = [None] * len(self.biases)

        for layer_idx in reversed(range(len(self.weights))):
            a_prev = activations[layer_idx]
            grad_w[layer_idx] = a_prev.T @ delta
            grad_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)
            if layer_idx > 0:
                da_prev = delta @ self.weights[layer_idx].T
                relu_mask = (pre[layer_idx - 1] > 0).astype(np.float64)
                delta = da_prev * relu_mask

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

        return loss

    def expected_mae_batch(
        self,
        states: np.ndarray,
        errors: np.ndarray,
    ) -> float:
        """
        One SGD step minimizing E_{a~pi}[err(s,a)] = sum_a pi(a|s) err(s,a).

        Aligns the softmax policy with mean absolute error on the training reactions
        (differentiable; complements hard cross-entropy warmup).
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        bsz = states.shape[0]
        logits, activations, pre = self._forward(states, return_cache=True)
        probs = _softmax(logits)
        exp_err = (probs * errors).sum(axis=1, keepdims=True)
        diff = errors - exp_err
        d_logits = probs * diff / max(1, bsz)

        loss = float(exp_err.mean())

        delta = d_logits
        grad_w: List[Optional[np.ndarray]] = [None] * len(self.weights)
        grad_b: List[Optional[np.ndarray]] = [None] * len(self.biases)

        for layer_idx in reversed(range(len(self.weights))):
            a_prev = activations[layer_idx]
            grad_w[layer_idx] = a_prev.T @ delta
            grad_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)
            if layer_idx > 0:
                da_prev = delta @ self.weights[layer_idx].T
                relu_mask = (pre[layer_idx - 1] > 0).astype(np.float64)
                delta = da_prev * relu_mask

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

        return loss

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "state_dim": self.state_dim,
                    "n_actions": self.n_actions,
                    "layer_sizes": self.layer_sizes,
                    "weights": self.weights,
                    "biases": self.biases,
                    "learning_rate": self.learning_rate,
                    "entropy_coef": self.entropy_coef,
                },
                f,
            )
        logger.info("Saved policy to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ReinforcePolicyAgent":
        with open(path, "rb") as f:
            d = pickle.load(f)
        ag = cls(
            state_dim=d["state_dim"],
            n_actions=d["n_actions"],
            hidden_layers=tuple(d["layer_sizes"][:-1]),
            learning_rate=d.get("learning_rate", 0.002),
            entropy_coef=d.get("entropy_coef", 0.01),
        )
        ag.weights = d["weights"]
        ag.biases = d["biases"]
        return ag
