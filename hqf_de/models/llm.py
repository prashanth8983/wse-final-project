import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from typing import List, Optional
from dataclasses import dataclass
import logging

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class Expansion:
    text: str
    gaps: List[str]
    expansions: List[str]


class LLM:

    GAP_PROMPT = """Analyze this document and list semantic gaps (max 5):
{document}

Gaps:"""

    EXPAND_PROMPT = """Generate brief factual expansions for this document:
{document}

Gaps: {gaps}

Expansions:"""

    def __init__(self, model: str = None, device: str = None, quantize: bool = True):
        self.model_name = model or config.llm_model_name
        self.device = device or config.device
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        logger.info(f"Loading LLM: {self.model_name}")

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if self.quantize and self.device == "cuda" else None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=config.cache_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device in ["mps", "cuda"] else torch.float32
        device_map = {"": self.device} if self.device != "cuda" else "auto"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=config.cache_dir, quantization_config=quant_config, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
        except:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=config.cache_dir, torch_dtype=dtype, trust_remote_code=True)
            if self.device != "cpu":
                self.model = self.model.to(self.device)

        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=config.llm_max_new_tokens, temperature=config.llm_temperature, do_sample=True)
        self._loaded = True
        return self

    def _fmt(self, doc: str, template: str, **kw) -> str:
        content = template.format(document=doc, **kw)
        if "llama" in self.model_name.lower():
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return f"[INST] {content} [/INST]"

    def _gen(self, prompt: str) -> str:
        if not self._loaded:
            self.load()
        result = self.pipe(prompt, return_full_text=False, pad_token_id=self.tokenizer.pad_token_id)
        return result[0]["generated_text"].strip()

    def _parse(self, text: str) -> List[str]:
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                if line.startswith("-"):
                    line = line[1:].strip()
                if line and len(line) > 5:
                    items.append(line)
        return items[:5]

    def gaps(self, doc: str) -> List[str]:
        return self._parse(self._gen(self._fmt(doc, self.GAP_PROMPT)))

    def expand(self, doc: str, gaps: List[str] = None) -> List[str]:
        gaps_text = "\n".join(gaps) if gaps else "none"
        return self._parse(self._gen(self._fmt(doc, self.EXPAND_PROMPT, gaps=gaps_text)))

    def run(self, doc: str) -> Expansion:
        g = self.gaps(doc)
        e = self.expand(doc, g)
        return Expansion(text=doc, gaps=g, expansions=e)

    def unload(self):
        if self.model:
            del self.model, self.tokenizer, self.pipe
            self.model = self.tokenizer = self.pipe = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
