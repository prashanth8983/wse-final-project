import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import logging

from ..config import config

logger = logging.getLogger(__name__)


class Embedder:

    def __init__(self, model: str = None, device: str = None):
        self.model_name = model or config.embedding_model
        self.device = device or config.device
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        logger.info(f"Loading embedder: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, cache_folder=str(config.cache_dir), device=self.device)
        self._loaded = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._loaded:
            self.load()
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def sim(self, texts1: List[str], texts2: List[str] = None) -> np.ndarray:
        e1 = self.encode(texts1)
        if texts2 is None:
            return cosine_similarity(e1)
        return cosine_similarity(e1, self.encode(texts2))

    def dedup(self, texts: List[str], threshold: float = 0.85) -> Tuple[List[str], List[int]]:
        if len(texts) <= 1:
            return texts, list(range(len(texts)))
        sim = self.sim(texts)
        kept = []
        indices = []
        removed = set()
        for i in range(len(texts)):
            if i in removed:
                continue
            kept.append(texts[i])
            indices.append(i)
            for j in range(i + 1, len(texts)):
                if sim[i, j] >= threshold:
                    removed.add(j)
        return kept, indices

    def dedup_vs_doc(self, doc: str, expansions: List[str], threshold: float = 0.85) -> List[str]:
        if not expansions:
            return []
        sims = self.sim(expansions, [doc]).flatten()
        return [e for e, s in zip(expansions, sims) if s < threshold]

    def select(self, expansions: List[str], n: int = 5, doc: str = None) -> List[str]:
        if len(expansions) <= n:
            return expansions
        embs = self.encode(expansions)
        rel = self.sim(expansions, [doc]).flatten() if doc else np.ones(len(expansions))
        selected = []
        indices = []
        for _ in range(n):
            best_idx, best_score = -1, -float('inf')
            for i in range(len(expansions)):
                if i in indices:
                    continue
                div = 1 - max(cosine_similarity(embs[i:i+1], embs[indices]).flatten()) if indices else 1.0
                score = 0.5 * rel[i] + 0.5 * div
                if score > best_score:
                    best_score, best_idx = score, i
            if best_idx >= 0:
                selected.append(expansions[best_idx])
                indices.append(best_idx)
        return selected

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
