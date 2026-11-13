from __future__ import annotations
import json
import math
import os
import re
from typing import Sequence
FEATURE_NAMES: tuple[str, ...] = ('log_len_chars', 'log_len_words', 'frac_digits', 'n_braces', 'n_dollar_signs', 'n_operators', 'n_keywords_easy', 'n_keywords_hard', 'max_brace_depth', 'log_max_number')
_EASY_KW = ('evaluate', 'compute', 'simplify', 'calculate', 'what is', 'value of', 'answer', 'round')
_HARD_KW = ('prove', 'show that', 'find all', 'how many', 'minimum', 'maximum', 'least', 'greatest', 'smallest', 'largest')

def _max_brace_depth(s: str) -> int:
    depth, best = (0, 0)
    for c in s:
        if c == '{':
            depth += 1
            best = max(best, depth)
        elif c == '}':
            depth = max(0, depth - 1)
    return best

def extract_features(question: str) -> list[float]:
    q = question or ''
    ql = q.lower()
    n_chars = max(1, len(q))
    words = re.findall('[a-zA-Z]+', q)
    n_words = max(1, len(words))
    digits = sum((c.isdigit() for c in q))
    numbers = re.findall('\\d+', q)
    max_num = max((int(x) for x in numbers if len(x) <= 9), default=0)
    n_easy = sum((ql.count(k) for k in _EASY_KW))
    n_hard = sum((ql.count(k) for k in _HARD_KW))
    n_ops = sum((q.count(c) for c in '+-*/=^'))
    return [math.log(n_chars), math.log(n_words), digits / n_chars, float(q.count('{')), float(q.count('$')), float(n_ops), float(n_easy), float(n_hard), float(_max_brace_depth(q)), math.log(1 + max_num)]

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

class DSCPrior:

    def __init__(self, weights: Sequence[float] | None=None, bias: float=0.0, feature_mean: Sequence[float] | None=None, feature_std: Sequence[float] | None=None):
        d = len(FEATURE_NAMES)
        self.weights: list[float] = list(weights) if weights is not None else [0.0] * d
        self.bias: float = float(bias)
        self.feature_mean: list[float] = list(feature_mean) if feature_mean is not None else [0.0] * d
        self.feature_std: list[float] = list(feature_std) if feature_std is not None else [1.0] * d
        if len(self.weights) != d or len(self.feature_mean) != d or len(self.feature_std) != d:
            raise ValueError('DSCPrior dimension mismatch with FEATURE_NAMES')

    def _normalise(self, feats: Sequence[float]) -> list[float]:
        return [(f - m) / (s if s > 1e-09 else 1.0) for f, m, s in zip(feats, self.feature_mean, self.feature_std)]

    def predict(self, question: str) -> float:
        feats = self._normalise(extract_features(question))
        z = self.bias + sum((w * f for w, f in zip(self.weights, feats)))
        return _sigmoid(z)

    def to_json(self) -> dict:
        return {'feature_names': list(FEATURE_NAMES), 'weights': self.weights, 'bias': self.bias, 'feature_mean': self.feature_mean, 'feature_std': self.feature_std}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'DSCPrior':
        with open(path) as f:
            d = json.load(f)
        names = d.get('feature_names')
        if names is not None and tuple(names) != FEATURE_NAMES:
            raise ValueError(f'Saved DSCPrior feature_names {names} do not match current FEATURE_NAMES {FEATURE_NAMES}; retrain.')
        return cls(weights=d['weights'], bias=d.get('bias', 0.0), feature_mean=d.get('feature_mean'), feature_std=d.get('feature_std'))

    @classmethod
    def fit(cls, X: Sequence[Sequence[float]], y: Sequence[int], *, lr: float=0.05, n_iter: int=2000, l2: float=0.001) -> 'DSCPrior':
        d = len(FEATURE_NAMES)
        n = len(X)
        if n == 0:
            raise ValueError('empty training set')
        if any((len(x) != d for x in X)):
            raise ValueError(f'all feature vectors must have dim {d}')
        mean = [sum((x[j] for x in X)) / n for j in range(d)]
        var = [sum(((x[j] - mean[j]) ** 2 for x in X)) / n for j in range(d)]
        std = [math.sqrt(v) if v > 1e-09 else 1.0 for v in var]
        Xn = [[(x[j] - mean[j]) / std[j] for j in range(d)] for x in X]
        w = [0.0] * d
        b = 0.0
        for _ in range(n_iter):
            grad_w = [0.0] * d
            grad_b = 0.0
            for xi, yi in zip(Xn, y):
                z = b + sum((w[j] * xi[j] for j in range(d)))
                p = _sigmoid(z)
                err = p - yi
                for j in range(d):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(d):
                grad_w[j] = grad_w[j] / n + l2 * w[j]
            grad_b = grad_b / n
            for j in range(d):
                w[j] -= lr * grad_w[j]
            b -= lr * grad_b
        return cls(weights=w, bias=b, feature_mean=mean, feature_std=std)
