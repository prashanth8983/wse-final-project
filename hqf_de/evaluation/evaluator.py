from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import subprocess
import time
import logging

from .metrics import Metrics, MetricResult
from ..pipeline.indexer_bridge import Bridge
from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    method: str
    metrics: MetricResult
    per_query: List[Dict[str, float]] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    avg_latency: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)


class Evaluator:

    def __init__(self, data_dir: Path = None, output_dir: Path = None, indexer_path: Path = None):
        self.data_dir = Path(data_dir or config.data_dir)
        self.output_dir = Path(output_dir or config.output_dir)
        self.indexer_path = indexer_path or (config.project_root / "indexer")
        self.bridge = Bridge(self.data_dir, self.output_dir)
        self.queries_path = self.data_dir / "queries.dev.tsv"
        self.qrels_path = self.data_dir / "qrels.dev.tsv"

    def load_queries(self, path: Path = None, limit: int = None) -> Dict[str, str]:
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

    def load_qrels(self, path: Path = None, qids: set = None) -> Dict[str, Dict[str, int]]:
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
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][pid] = rel
        return qrels

    def index(self, tsv: Path) -> Tuple[bool, float]:
        binary = self.indexer_path / "indexer"
        if not binary.exists():
            return False, 0.0
        start = time.time()
        try:
            result = subprocess.run([str(binary), str(tsv)], cwd=str(self.indexer_path), capture_output=True, text=True, timeout=3600)
            elapsed = time.time() - start
            return result.returncode == 0, elapsed
        except:
            return False, 0.0

    def evaluate(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], name: str = "unknown", ks: List[int] = [10, 100, 1000]) -> EvalResult:
        all_results = []
        latencies = []
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
        return EvalResult(
            method=name,
            metrics=Metrics.aggregate(all_results),
            per_query=all_results,
            latencies=latencies,
            avg_latency=sum(latencies) / len(latencies) if latencies else 0.0
        )

    def compare(self, results: List, num_queries: int = 100) -> Dict[str, EvalResult]:
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

    def save(self, results: Dict[str, EvalResult], path: Path = None):
        path = path or self.output_dir / "results.json"
        data = {}
        for method, r in results.items():
            data[method] = {
                "metrics": {"ndcg@10": r.metrics.ndcg_at_10, "ndcg@100": r.metrics.ndcg_at_100,
                           "recall@10": r.metrics.recall_at_10, "recall@100": r.metrics.recall_at_100,
                           "recall@1000": r.metrics.recall_at_1000, "mrr@10": r.metrics.mrr_at_10, "map": r.metrics.map_score},
                "num_queries": r.metrics.num_queries,
                "avg_latency": r.avg_latency,
                "stats": r.stats
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def report(self, results: Dict[str, EvalResult]) -> str:
        lines = ["# Evaluation Report", "", "| Method | nDCG@10 | Recall@100 | Recall@1000 | MRR@10 |",
                 "|--------|---------|------------|-------------|--------|"]
        for method, r in results.items():
            m = r.metrics
            lines.append(f"| {method} | {m.ndcg_at_10:.4f} | {m.recall_at_100:.4f} | {m.recall_at_1000:.4f} | {m.mrr_at_10:.4f} |")
        return "\n".join(lines)
