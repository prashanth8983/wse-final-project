import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List, Tuple
from dataclasses import dataclass
import logging

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class NLIResult:
    hypothesis: str
    entailment: float
    valid: bool


class NLI:

    def __init__(self, model: str = None, device: str = None, threshold: float = 0.9):
        self.model_name = model or config.nli_model
        self.device = device or config.device
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        logger.info(f"Loading NLI: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=config.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=config.cache_dir)
        if self.device in ["mps", "cuda"]:
            self.model = self.model.to(self.device)
        self.model.eval()
        device_id = "mps" if self.device == "mps" else (0 if self.device == "cuda" else -1)
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=device_id, top_k=None)
        self._loaded = True
        return self

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        if not self._loaded:
            self.load()
        try:
            results = self.pipe(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
            scores = {r["label"].lower(): r["score"] for r in results}
            ent = scores.get("entailment", 0.0)
            return NLIResult(hypothesis=hypothesis, entailment=ent, valid=ent >= self.threshold)
        except:
            return NLIResult(hypothesis=hypothesis, entailment=0.0, valid=False)

    def validate(self, doc: str, expansions: List[str]) -> Tuple[List[str], List[NLIResult]]:
        if not self._loaded:
            self.load()
        valid = []
        results = []
        for exp in expansions:
            r = self.check(doc, exp)
            results.append(r)
            if r.valid:
                valid.append(exp)
        return valid, results

    def unload(self):
        if self.model:
            del self.model, self.tokenizer, self.pipe
            self.model = self.tokenizer = self.pipe = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
