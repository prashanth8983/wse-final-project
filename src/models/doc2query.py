import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Doc2Query:
    def __init__(self, model="castorini/doc2query-t5-base-msmarco", device="cuda", num_queries=5):
        self.model_name = model
        self.device = device
        self.num_queries = num_queries
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model:
            return self
        print(f"Loading Doc2Query: {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        if self.device in ["mps", "cuda"]:
            self.model = self.model.to(self.device)
        self.model.eval()
        return self

    def generate(self, doc, n=None):
        if not self.model:
            self.load()
        n = n or self.num_queries
        inputs = self.tokenizer(doc, max_length=512, truncation=True, return_tensors="pt")
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
