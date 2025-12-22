from ..models.llm import LLM
from ..models.nli import NLI
from ..models.doc2query import Doc2Query
from ..models.embeddings import Embedder
from .combiner import Combiner


class Expander:
    def __init__(self, use_llm=True, use_nli=True, use_d2q=True, device="cuda"):
        self.device = device
        self.llm = LLM(device=device) if use_llm else None
        self.nli = NLI(device=device) if use_nli else None
        self.d2q = Doc2Query(device=device) if use_d2q else None
        self.combiner = Combiner(Embedder(device=device))

    def load(self):
        if self.llm: self.llm.load()
        if self.nli: self.nli.load()
        if self.d2q: self.d2q.load()
        self.combiner.load()
        return self

    def expand(self, doc_id, doc):
        result = {"doc_id": doc_id, "original": doc, "gaps": [], "raw_expansions": [],
                  "valid_expansions": [], "queries": [], "final": [], "expanded": doc}

        raw = []
        if self.llm:
            try:
                exp = self.llm.run(doc)
                result["gaps"] = exp["gaps"]
                raw = exp["expansions"]
                result["raw_expansions"] = raw
            except Exception as e:
                print(f"LLM error: {e}")

        valid = raw
        if self.nli and raw:
            try:
                valid = self.nli.validate(doc, raw)
                result["valid_expansions"] = valid
            except Exception as e:
                print(f"NLI error: {e}")

        queries = []
        if self.d2q:
            try:
                queries = self.d2q.generate(doc)
                result["queries"] = queries
            except Exception as e:
                print(f"D2Q error: {e}")

        try:
            combined = self.combiner.combine(doc, valid, queries)
            result["final"] = combined["final"]
            result["expanded"] = combined["text"]
        except Exception as e:
            print(f"Combiner error: {e}")
            all_exp = valid + queries
            result["final"] = all_exp[:10]
            result["expanded"] = f"{doc} {' '.join(all_exp[:10])}"

        return result

    def d2q_only(self, doc_id, doc):
        if self.d2q:
            self.d2q.load()
            queries = self.d2q.generate(doc)
        else:
            queries = []
        return {"doc_id": doc_id, "original": doc, "queries": queries, "expanded": f"{doc} {' '.join(queries)}"}

    def unload(self):
        if self.llm: self.llm.unload()
        if self.nli: self.nli.unload()
        if self.d2q: self.d2q.unload()
        self.combiner.unload()

    def __enter__(self):
        return self.load()

    def __exit__(self, *args):
        self.unload()
