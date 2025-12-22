import torch
from transformers import pipeline


class NLI:
    def __init__(self, model="facebook/bart-large-mnli", device="cuda"):
        self.model_name = model
        self.device = device
        self.pipe = None

    def load(self):
        if self.pipe:
            return self
        print(f"Loading NLI: {self.model_name}")
        device_id = 0 if self.device == "cuda" else (-1 if self.device == "cpu" else "mps")
        self.pipe = pipeline("text-classification", model=self.model_name, device=device_id)
        return self

    def check(self, premise, hypothesis):
        if not self.pipe:
            self.load()
        try:
            result = self.pipe(f"{premise}</s></s>{hypothesis}", truncation=True)
            return result[0]["label"] != "contradiction"
        except:
            return False

    def validate(self, doc, expansions):
        if not self.pipe:
            self.load()
        return [e for e in expansions if self.check(doc, e)]

    def unload(self):
        if self.pipe:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
