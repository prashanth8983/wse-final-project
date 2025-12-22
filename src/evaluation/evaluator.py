from pathlib import Path
from dataclasses import dataclass, field
import json
import subprocess
import time

from .metrics import Metrics, MetricResult
from ..pipeline.indexer_bridge import Bridge
from ..config import config


@dataclass
class EvalResult:
    method: str
    metrics: MetricResult
    per_query: list = field(default_factory=list)
    latencies: list = field(default_factory=list)
    avg_latency: float = 0.0
    stats: dict = field(default_factory=dict)


class Evaluator:
    def __init__(self, data_dir=None, output_dir=None, indexer_path=None):
        self.data_dir = Path(data_dir or config.data_dir)
        self.output_dir = Path(output_dir or config.output_dir)
        self.indexer_path = indexer_path or (config.project_root / "indexer")
        self.bridge = Bridge(self.data_dir, self.output_dir)
        self.queries_path = self.data_dir / "queries.dev.tsv"
        self.qrels_path = self.data_dir / "qrels.dev.tsv"

    def load_queries(self, path=None, limit=None):
        path = path or self.queries_path
        if not path.exists():
            return {}
        queries = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]
                    if limit and len(queries) >= limit:
                        break
        return queries

    def load_qrels(self, path=None, qids=None):
        path = path or self.qrels_path
        if not path.exists():
            return {}
        qrels = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    qid, pid, rel = parts[0], parts[2], int(parts[3])
                    if qids and qid not in qids:
                        continue
                    qrels.setdefault(qid, {})[pid] = rel
        return qrels

    def index(self, tsv):
        binary = self.indexer_path / "indexer"
        if not binary.exists():
            return False, 0.0
        start = time.time()
        try:
            result = subprocess.run([str(binary), str(tsv)], cwd=str(self.indexer_path), capture_output=True, text=True, timeout=3600)
            return result.returncode == 0, time.time() - start
        except:
            return False, 0.0

    def evaluate(self, queries, qrels, name="unknown", ks=[10, 100, 1000]):
        all_results, latencies = [], []
        for qid, text in queries.items():
            if qid not in qrels:
                continue
            relevant = set(qrels[qid].keys())
            start = time.time()
            results = self.bridge.search(text, limit=max(ks))
            latencies.append((time.time() - start) * 1000)
            retrieved = [r.passage_id for r in results]
            relevances = [float(qrels[qid].get(r.passage_id, 0)) for r in results]
            all_results.append(Metrics.all(retrieved, relevances, relevant, ks))
        return EvalResult(method=name, metrics=Metrics.aggregate(all_results), per_query=all_results, latencies=latencies, avg_latency=sum(latencies) / len(latencies) if latencies else 0.0)

    def compare(self, results, num_queries=100):
        queries = self.load_queries(limit=num_queries)
        qrels = self.load_qrels(qids=set(queries.keys()))
        if not queries or not qrels:
            return {}
        paths = self.bridge.write_comparison(results)
        evals = {}
        for method, tsv in paths.items():
            ok, t = self.index(tsv)
            if not ok:
                continue
            ev = self.evaluate(queries, qrels, name=method)
            ev.stats["index_time"] = t
            evals[method] = ev
        return evals

    def save(self, results, path=None):
        path = path or self.output_dir / "results.json"
        data = {method: {"metrics": {"ndcg@10": r.metrics.ndcg_at_10, "recall@100": r.metrics.recall_at_100, "mrr@10": r.metrics.mrr_at_10}, "avg_latency": r.avg_latency} for method, r in results.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def report(self, results):
        lines = ["| Method | nDCG@10 | Recall@100 | MRR@10 |", "|--------|---------|------------|--------|"]
        for method, r in results.items():
            lines.append(f"| {method} | {r.metrics.ndcg_at_10:.4f} | {r.metrics.recall_at_100:.4f} | {r.metrics.mrr_at_10:.4f} |")
        return "\n".join(lines)
