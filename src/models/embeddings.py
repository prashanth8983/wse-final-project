import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Embedder:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.model_name = model
        self.device = device
        self.model = None

    def load(self):
        if self.model:
            return self
        print(f"Loading Embedder: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def encode(self, texts):
        if not self.model:
            self.load()
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def similarity(self, texts1, texts2=None):
        e1 = self.encode(texts1)
        return cosine_similarity(e1) if texts2 is None else cosine_similarity(e1, self.encode(texts2))

    def deduplicate(self, texts, threshold=0.85):
        if len(texts) <= 1:
            return texts
        sim = self.similarity(texts)
        kept, removed = [], set()
        for i in range(len(texts)):
            if i in removed:
                continue
            kept.append(texts[i])
            for j in range(i + 1, len(texts)):
                if sim[i, j] >= threshold:
                    removed.add(j)
        return kept

    def filter_similar_to_doc(self, doc, expansions, threshold=0.85):
        if not expansions:
            return []
        sims = self.similarity(expansions, [doc]).flatten()
        return [e for e, s in zip(expansions, sims) if s < threshold]

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
