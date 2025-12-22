import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

GAP_PROMPT = """Analyze this document and list semantic gaps (max 5):
{document}

Gaps:"""

EXPAND_PROMPT = """Generate brief factual expansions for this document:
{document}

Gaps: {gaps}

Expansions:"""


class LLM:
    def __init__(self, model="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda"):
        self.model_name = model
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipe = None

    def load(self):
        if self.model:
            return self
        print(f"Loading LLM: {self.model_name}")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if self.device == "cuda" else None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device in ["mps", "cuda"] else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=quant, torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else {"": self.device}, trust_remote_code=True
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer,
                            max_new_tokens=256, temperature=0.7, do_sample=True)
        return self

    def _format(self, doc, template, **kw):
        content = template.format(document=doc, **kw)
        if "llama" in self.model_name.lower():
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return f"[INST] {content} [/INST]"

    def _generate(self, prompt):
        if not self.model:
            self.load()
        result = self.pipe(prompt, return_full_text=False, pad_token_id=self.tokenizer.pad_token_id)
        return result[0]["generated_text"].strip()

    def _parse(self, text):
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line[0].isdigit():
                line = line.split(".", 1)[-1].strip()
            if line.startswith("-"):
                line = line[1:].strip()
            if len(line) > 5:
                items.append(line)
        return items[:5]

    def gaps(self, doc):
        return self._parse(self._generate(self._format(doc, GAP_PROMPT)))

    def expand(self, doc, gaps=None):
        gaps_text = "\n".join(gaps) if gaps else "none"
        return self._parse(self._generate(self._format(doc, EXPAND_PROMPT, gaps=gaps_text)))

    def run(self, doc):
        gaps = self.gaps(doc)
        return {"text": doc, "gaps": gaps, "expansions": self.expand(doc, gaps)}

    def unload(self):
        if self.model:
            del self.model, self.tokenizer, self.pipe
            self.model = self.tokenizer = self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
