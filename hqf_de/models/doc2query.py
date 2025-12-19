import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Optional
import logging

from ..config import config

logger = logging.getLogger(__name__)


class Doc2Query:

    def __init__(self, model: str = None, device: str = None, n: int = 5):
        self.model_name = model or config.doc2query_model
        self.device = device or config.device
        self.n = n
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        logger.info(f"Loading doc2query: {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, cache_dir=config.cache_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, cache_dir=config.cache_dir)
        if self.device in ["mps", "cuda"]:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        return self

    def gen(self, doc: str, n: int = None) -> List[str]:
        if not self._loaded:
            self.load()
        n = n or self.n
        inputs = self.tokenizer(doc, max_length=config.max_doc_length, truncation=True, return_tensors="pt")
        if self.device in ["mps", "cuda"]:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=64, do_sample=True, top_k=10, num_return_sequences=n)
        queries = []
        for out in outputs:
            q = self.tokenizer.decode(out, skip_special_tokens=True).strip()
            if q and q not in queries:
                queries.append(q)
        return queries

    def unload(self):
        if self.model:
            del self.model, self.tokenizer
            self.model = self.tokenizer = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
