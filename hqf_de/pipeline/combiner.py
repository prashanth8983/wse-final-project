from typing import List, Dict, Any
from dataclasses import dataclass, field
import logging

from ..models.embeddings import Embedder

logger = logging.getLogger(__name__)

GENERIC = {"information", "details", "things", "stuff", "content", "topic", "subject", "matter", "example", "case", "way", "method", "people", "time", "place", "thing"}


@dataclass
class Combined:
    original: str
    semantic: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    final: List[str] = field(default_factory=list)
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class Combiner:

    def __init__(self, embedder: Embedder = None, threshold: float = 0.85, max_exp: int = 10):
        self.embedder = embedder or Embedder()
        self.threshold = threshold
        self.max_exp = max_exp
        self._loaded = False

    def _load(self):
        if not self._loaded:
            self.embedder.load()
            self._loaded = True

    def _filter(self, expansions: List[str], doc: str = None) -> List[str]:
        out = []
        for e in expansions:
            e = e.strip()
            if not e or len(e.split()) < 3 or len(e.split()) > 50:
                continue
            words = e.lower().split()
            if sum(1 for w in words if w in GENERIC) / len(words) > 0.5:
                continue
            if doc and e.lower() in doc.lower():
                continue
            out.append(e)
        return out

    def _dedup(self, expansions: List[str], doc: str = None) -> List[str]:
        if len(expansions) <= 1:
            return expansions
        self._load()
        deduped, _ = self.embedder.dedup(expansions, self.threshold)
        if doc:
            deduped = self.embedder.dedup_vs_doc(doc, deduped, self.threshold)
        return deduped

    def combine(self, doc: str, semantic: List[str], queries: List[str]) -> Combined:
        self._load()
        sem = self._filter(semantic, doc)
        q = self._filter(queries, doc)
        all_exp = sem + q
        deduped = self._dedup(all_exp, doc)
        final = self.embedder.select(deduped, self.max_exp, doc) if len(deduped) > self.max_exp else deduped[:self.max_exp]
        text = f"{doc} {' '.join(final)}"
        return Combined(original=doc, semantic=sem, queries=q, final=final, text=text, meta={"n_sem": len(sem), "n_q": len(q), "n_final": len(final)})

    def unload(self):
        if self._loaded:
            self.embedder.unload()
            self._loaded = False
