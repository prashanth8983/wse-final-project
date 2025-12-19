from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

from ..models.llm import LLM
from ..models.nli import NLI
from ..models.doc2query import Doc2Query
from ..models.embeddings import Embedder
from .combiner import Combiner
from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class Result:
    doc_id: str
    original: str
    expanded: str
    gaps: List[str] = field(default_factory=list)
    raw: List[str] = field(default_factory=list)
    valid: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    final: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class Expander:

    def __init__(self, use_llm: bool = True, use_nli: bool = True, use_d2q: bool = True, device: str = None):
        self.device = device or config.device
        self.llm = LLM(device=self.device) if use_llm else None
        self.nli = NLI(device=self.device) if use_nli else None
        self.d2q = Doc2Query(device=self.device) if use_d2q else None
        self.combiner = Combiner(Embedder(device=self.device))
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        if self.llm:
            self.llm.load()
        if self.nli:
            self.nli.load()
        if self.d2q:
            self.d2q.load()
        self._loaded = True
        return self

    def expand(self, doc_id: str, doc: str) -> Result:
        if not self._loaded:
            self.load()

        result = Result(doc_id=doc_id, original=doc, expanded=doc)

        raw = []
        if self.llm:
            try:
                exp = self.llm.run(doc)
                result.gaps = exp.gaps
                raw = exp.expansions
                result.raw = raw
            except Exception as e:
                logger.error(f"LLM error: {e}")

        valid = raw
        if self.nli and raw:
            try:
                valid, _ = self.nli.validate(doc, raw)
                result.valid = valid
                result.rejected = [e for e in raw if e not in valid]
            except Exception as e:
                logger.error(f"NLI error: {e}")

        queries = []
        if self.d2q:
            try:
                queries = self.d2q.gen(doc)
                result.queries = queries
            except Exception as e:
                logger.error(f"D2Q error: {e}")

        try:
            combined = self.combiner.combine(doc, valid, queries)
            result.final = combined.final
            result.expanded = combined.text
            result.meta = combined.meta
        except Exception as e:
            logger.error(f"Combiner error: {e}")
            all_exp = valid + queries
            result.final = all_exp[:10]
            result.expanded = f"{doc} {' '.join(all_exp[:10])}"

        return result

    def d2q_only(self, doc_id: str, doc: str) -> Result:
        if self.d2q and not self.d2q._loaded:
            self.d2q.load()
        queries = self.d2q.gen(doc) if self.d2q else []
        return Result(doc_id=doc_id, original=doc, expanded=f"{doc} {' '.join(queries)}", queries=queries, final=queries, meta={"method": "d2q_only"})

    def unload(self):
        if self.llm:
            self.llm.unload()
        if self.nli:
            self.nli.unload()
        if self.d2q:
            self.d2q.unload()
        self.combiner.unload()
        self._loaded = False

    def __enter__(self):
        return self.load()

    def __exit__(self, *args):
        self.unload()
