#!/usr/bin/env python3
import h5py
import numpy as np
import faiss
from tqdm import tqdm
import subprocess
import os

DATA_DIR = "data"
RESULTS_DIR = "results"

VARIANTS = {
    "original": "embeddings_original.h5",
    "expanded": "embeddings_expanded.h5",
    "validated": "embeddings_validated.h5",
    "doc2query": "embeddings_doc2query.h5",
}
QUERY_FILE = "msmarco_queries_dev_eval_embeddings.h5"
QRELS = {"dev": "qrels.dev.tsv", "eval_2019": "qrels.eval.one.tsv", "eval_2020": "qrels.eval.two.tsv"}

def load_h5_embeddings(file_path):
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f['id']).astype(str)
        ids = np.array([x.decode() if isinstance(x, bytes) else x.replace("b'", "").replace("'", "") for x in ids])
        embeddings = np.array(f['embedding']).astype(np.float32)
    return ids, embeddings

def run_hnsw_retrieval(passage_file, query_file, output_file, variant_name):
    print(f"\n{'='*60}\nRunning HNSW: {variant_name}\n{'='*60}")

    passage_ids, passage_embeddings = load_h5_embeddings(passage_file)
    query_ids, query_embeddings = load_h5_embeddings(query_file)
    print(f"Passages: {len(passage_ids)}, Queries: {len(query_ids)}")

    faiss.normalize_L2(passage_embeddings)
    faiss.normalize_L2(query_embeddings)

    dim = passage_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 256
    faiss.omp_set_num_threads(faiss.omp_get_max_threads())
    index.add(passage_embeddings)

    K, batch_size = 100, 100
    with open(output_file, "w") as f:
        for start in tqdm(range(0, len(query_embeddings), batch_size), desc="Retrieving"):
            batch = query_embeddings[start:start+batch_size]
            distances, labels = index.search(batch, K)
            for qi, qid in enumerate(query_ids[start:start+batch_size]):
                qid = str(qid).strip().replace("b'", "").replace("'", "")
                for rank, (pid_idx, score) in enumerate(zip(labels[qi], distances[qi])):
                    f.write(f"{qid} Q0 {passage_ids[pid_idx]} {rank+1} {score:.6f} hnsw_{variant_name}\n")
    print(f"Saved: {output_file}")
    return output_file

def run_trec_eval(qrels_file, run_file):
    cmd = ["./trec_eval_bin", "-m", "recip_rank.10", "-m", "recall.100", "-m", "ndcg_cut.10", "-m", "ndcg_cut.100", qrels_file, run_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except FileNotFoundError:
        return "trec_eval not found"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    query_file = os.path.join(DATA_DIR, QUERY_FILE)
    all_results = {}

    for variant_name, emb_file in VARIANTS.items():
        passage_file = os.path.join(DATA_DIR, emb_file)
        output_file = os.path.join(RESULTS_DIR, f"run_hnsw_{variant_name}.txt")
        run_hnsw_retrieval(passage_file, query_file, output_file, variant_name)
        all_results[variant_name] = output_file

    print(f"\n{'='*60}\nEVALUATION RESULTS\n{'='*60}")
    for variant_name, run_file in all_results.items():
        print(f"\n--- {variant_name.upper()} ---")
        for qrels_name, qrels_file in QRELS.items():
            print(f"\n{qrels_name}:")
            print(run_trec_eval(os.path.join(DATA_DIR, qrels_file), run_file))

if __name__ == "__main__":
    main()
