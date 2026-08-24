import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]:
        ...


class HashEmbedder:
    name = "hash256"
    dim = 256

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            index = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def default_embedder() -> Embedder:
    return HashEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
