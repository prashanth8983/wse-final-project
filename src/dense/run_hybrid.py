#!/usr/bin/env python3
import os
import subprocess
from collections import defaultdict

K = 60  # RRF constant
RESULTS_DIR = "results"
VARIANTS = ["original", "expanded", "validated", "doc2query"]
QRELS = {"trec2019": "data/qrels.eval.one.tsv", "trec2020": "data/qrels.eval.two.tsv", "dev": "data/qrels.dev.trec.tsv"}

def load_qrels_qids():
    qids = set()
    for qrels_file in QRELS.values():
        if os.path.exists(qrels_file):
            with open(qrels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        qids.add(parts[0])
    return qids

def load_run_file(filepath, qid_filter=None):
    results = defaultdict(list)
    if not os.path.exists(filepath):
        return results
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid = parts[0]
                if qid_filter and qid not in qid_filter:
                    continue
                results[qid].append((parts[2], int(parts[3]), float(parts[4])))
    return results

def reciprocal_rank_fusion(run_files, k=60, qid_filter=None):
    all_runs = [load_run_file(f, qid_filter) for f in run_files if os.path.exists(f)]
    if not all_runs:
        return {}

    all_qids = set()
    for run in all_runs:
        all_qids.update(run.keys())

    fused_results = {}
    for qid in all_qids:
        doc_scores = defaultdict(float)
        for run in all_runs:
            if qid in run:
                for docid, rank, _ in run[qid]:
                    doc_scores[docid] += 1.0 / (k + rank)
        fused_results[qid] = sorted(doc_scores.items(), key=lambda x: -x[1])[:1000]
    return fused_results

def write_run_file(fused_results, output_path, run_name="hybrid"):
    with open(output_path, 'w') as f:
        for qid in sorted(fused_results.keys(), key=lambda x: int(x) if x.isdigit() else x):
            for rank, (docid, score) in enumerate(fused_results[qid], 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")

def run_trec_eval(qrels_file, run_file):
    cmd = ["./trec_eval_bin", "-m", "recip_rank.10", "-m", "recall.100", "-m", "ndcg_cut.10", qrels_file, run_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    except:
        return ""

def main():
    print("=" * 70)
    print("HYBRID RETRIEVAL (BM25 + HNSW with RRF)")
    print("=" * 70)

    qid_filter = load_qrels_qids()
    all_results = {}

    for variant in VARIANTS:
        print(f"\n--- {variant.upper()} ---")
        bm25_file = f"{RESULTS_DIR}/run_bm25_cpp_{variant}.txt"
        hnsw_file = f"{RESULTS_DIR}/run_hnsw_{variant}.txt"

        if not os.path.exists(bm25_file) or not os.path.exists(hnsw_file):
            print(f"  Missing files, skipping")
            continue

        fused = reciprocal_rank_fusion([bm25_file, hnsw_file], k=K, qid_filter=qid_filter)
        output_file = f"{RESULTS_DIR}/run_hybrid_{variant}.txt"
        write_run_file(fused, output_file, run_name=f"hybrid_{variant}")
        print(f"  Saved: {output_file}")

        variant_results = {}
        for qrels_name, qrels_file in QRELS.items():
            output = run_trec_eval(qrels_file, output_file)
            metrics = {}
            for line in output.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 3:
                    metrics[parts[0]] = float(parts[2])
            variant_results[qrels_name] = metrics
            print(f"  {qrels_name}: MRR={metrics.get('recip_rank_10', 0):.4f}")
        all_results[variant] = variant_results

    print("\n" + "=" * 70)
    print("Done!")

if __name__ == "__main__":
    main()
