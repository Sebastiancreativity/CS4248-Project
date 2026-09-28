"""Shared metrics and a small dependency-free logistic selector."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable

import numpy as np


FEATURE_NAMES = [
    "score1",
    "score2",
    "score2_minus_score1",
    "source_to_pass1_edit_ratio",
    "pass1_to_pass2_edit_ratio",
    "source_to_pass2_edit_ratio",
    "pass1_length_ratio",
    "pass2_length_ratio",
    "stable_after_pass1",
]


def tokens(text: str) -> list[str]:
    return text.split()


def edit_distance(left: str, right: str) -> int:
    """Token-level Levenshtein distance."""
    a, b = tokens(left), tokens(right)
    previous = list(range(len(b) + 1))
    for i, left_token in enumerate(a, 1):
        current = [i]
        for j, right_token in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def edit_ratio(left: str, right: str) -> float:
    denominator = max(len(tokens(left)), len(tokens(right)), 1)
    return edit_distance(left, right) / denominator


def copy_ratio(source: str, candidate: str) -> float:
    candidate_tokens = tokens(candidate)
    if not candidate_tokens:
        return 0.0
    blocks = SequenceMatcher(None, tokens(source), candidate_tokens).get_matching_blocks()
    return sum(block.size for block in blocks) / len(candidate_tokens)


def make_features(source: str, pass1: str, pass2: str, score1: float, score2: float) -> list[float]:
    source_length = max(len(tokens(source)), 1)
    return [
        score1,
        score2,
        score2 - score1,
        edit_ratio(source, pass1),
        edit_ratio(pass1, pass2),
        edit_ratio(source, pass2),
        len(tokens(pass1)) / source_length,
        len(tokens(pass2)) / source_length,
        float(pass1 == pass2),
    ]


def oracle_label(pass1: str, pass2: str, reference: str) -> int:
    """Prefer pass 1 on ties to avoid an unnecessary extra rewrite."""
    return int(edit_distance(pass2, reference) < edit_distance(pass1, reference))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic(
    rows: Iterable[Iterable[float]],
    labels: Iterable[int],
    learning_rate: float = 0.08,
    steps: int = 2500,
    l2: float = 0.01,
) -> dict[str, object]:
    x = np.asarray(list(rows), dtype=np.float64)
    y = np.asarray(list(labels), dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or len(x) != len(y):
        raise ValueError("Need a non-empty feature matrix aligned with labels")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    z = np.column_stack([np.ones(len(z)), z])
    weights = np.zeros(z.shape[1], dtype=np.float64)
    positive_rate = float(y.mean())
    if 0.0 < positive_rate < 1.0:
        weights[0] = np.log(positive_rate / (1.0 - positive_rate))
    for _ in range(steps):
        probabilities = sigmoid(z @ weights)
        gradient = z.T @ (probabilities - y) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return {
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "threshold": 0.5,
        "positive_rate": positive_rate,
    }


def predict_probability(model: dict[str, object], features: Iterable[float]) -> float:
    values = np.asarray(list(features), dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    normalized = (values - mean) / scale
    vector = np.concatenate([[1.0], normalized])
    return float(sigmoid(np.asarray([vector @ weights]))[0])

