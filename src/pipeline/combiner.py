from ..models.embeddings import Embedder

GENERIC = {"information", "details", "things", "stuff", "content", "topic", "subject", "example", "case", "way", "method", "people", "time", "place", "thing"}


class Combiner:
    def __init__(self, embedder=None, threshold=0.85, max_expansions=10):
        self.embedder = embedder or Embedder()
        self.threshold = threshold
        self.max_expansions = max_expansions

    def load(self):
        self.embedder.load()
        return self

    def filter_expansions(self, expansions, doc=None):
        out = []
        for e in expansions:
            e = e.strip()
            words = e.lower().split()
            if len(words) < 3 or len(words) > 50:
                continue
            if sum(1 for w in words if w in GENERIC) / len(words) > 0.5:
                continue
            if doc and e.lower() in doc.lower():
                continue
            out.append(e)
        return out

    def deduplicate(self, expansions, doc=None):
        if len(expansions) <= 1:
            return expansions
        self.embedder.load()
        deduped = self.embedder.deduplicate(expansions, self.threshold)
        if doc:
            deduped = self.embedder.filter_similar_to_doc(doc, deduped, self.threshold)
        return deduped

    def combine(self, doc, semantic_exp, query_exp):
        self.embedder.load()
        sem = self.filter_expansions(semantic_exp, doc)
        queries = self.filter_expansions(query_exp, doc)
        deduped = self.deduplicate(sem + queries, doc)
        final = deduped[:self.max_expansions]
        return {"original": doc, "semantic": sem, "queries": queries, "final": final, "text": f"{doc} {' '.join(final)}"}

    def unload(self):
        self.embedder.unload()
