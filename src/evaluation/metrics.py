import numpy as np
from typing import List, Dict, Set
from dataclasses import dataclass

@dataclass
class MetricResult:
    ndcg_at_10: float = 0.0
    ndcg_at_100: float = 0.0
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    recall_at_1000: float = 0.0
    mrr_at_10: float = 0.0
    map_score: float = 0.0
    num_queries: int = 0


class Metrics:

    @staticmethod
    def dcg(relevances: List[float], k: int) -> float:
        relevances = np.array(relevances[:k])
        if len(relevances) == 0:
            return 0.0
        discounts = np.log2(np.arange(len(relevances)) + 2)
        return np.sum(relevances / discounts)

    @staticmethod
    def ndcg(relevances: List[float], k: int, ideal: List[float] = None) -> float:
        dcg = Metrics.dcg(relevances, k)
        if ideal is None:
            ideal = sorted(relevances, reverse=True)
        idcg = Metrics.dcg(ideal, k)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def recall(retrieved: List[str], relevant: Set[str], k: int) -> float:
        if not relevant:
            return 0.0
        hits = len(set(retrieved[:k]) & relevant)
        return hits / len(relevant)

    @staticmethod
    def mrr(retrieved: List[str], relevant: Set[str], k: int = None) -> float:
        if k:
            retrieved = retrieved[:k]
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ap(retrieved: List[str], relevant: Set[str]) -> float:
        if not relevant:
            return 0.0
        hits = 0
        total = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                hits += 1
                total += hits / (i + 1)
        if hits == 0:
            return 0.0
        return total / len(relevant)

    @classmethod
    def all(cls, retrieved: List[str], relevances: List[float], relevant: Set[str], ks: List[int] = [10, 100, 1000]) -> Dict[str, float]:
        results = {}
        for k in ks:
            results[f"ndcg@{k}"] = cls.ndcg(relevances, k)
            results[f"recall@{k}"] = cls.recall(retrieved, relevant, k)
        results["mrr@10"] = cls.mrr(retrieved, relevant, k=10)
        results["map"] = cls.ap(retrieved, relevant)
        return results

    @classmethod
    def aggregate(cls, all_results: List[Dict[str, float]]) -> MetricResult:
        if not all_results:
            return MetricResult()
        agg = {}
        for key in all_results[0].keys():
            agg[key] = np.mean([r[key] for r in all_results])
        return MetricResult(
            ndcg_at_10=agg.get("ndcg@10", 0.0),
            ndcg_at_100=agg.get("ndcg@100", 0.0),
            recall_at_10=agg.get("recall@10", 0.0),
            recall_at_100=agg.get("recall@100", 0.0),
            recall_at_1000=agg.get("recall@1000", 0.0),
            mrr_at_10=agg.get("mrr@10", 0.0),
            map_score=agg.get("map", 0.0),
            num_queries=len(all_results)
        )
