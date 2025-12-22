import h5py
import numpy as np
import faiss
from tqdm import tqdm

BM25_FILES = ["queries.dev_results.txt", "queries.eval_results.txt"]
PASSAGE_H5 = "ms_marco/msmarco_passages_embeddings_subset.h5"
QUERY_H5 = "ms_marco/msmarco_queries_dev_eval_embeddings.h5"
TOP_K_BM25 = 1000
TOP_K_FINAL = 1000

def load_h5_embeddings(file_path, id_key='id', embedding_key='embedding'):
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(str)
        embeddings = np.array(f[embedding_key]).astype(np.float32)
    return ids, embeddings

print("Loading passage embeddings...")
passage_ids, passage_embeddings = load_h5_embeddings(PASSAGE_H5)
faiss.normalize_L2(passage_embeddings)

print("Loading query embeddings...")
query_ids, query_embeddings = load_h5_embeddings(QUERY_H5)
faiss.normalize_L2(query_embeddings)

def load_bm25_results(file_path, top_k=TOP_K_BM25):
    query_candidates = {}
    with open(file_path) as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split()
            rank = int(rank)
            if qid not in query_candidates:
                query_candidates[qid] = []
            if rank <= top_k:
                query_candidates[qid].append(pid)
    return query_candidates

def hybrid_rerank(query_embeddings, query_ids, passage_embeddings, passage_ids, bm25_results):
    pid_to_idx = {pid: i for i, pid in enumerate(passage_ids)}
    reranked = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Reranking")):
        candidates = bm25_results.get(str(qid), [])
        if not candidates:
            continue
        candidate_indices = [pid_to_idx[pid] for pid in candidates]
        candidate_emb = passage_embeddings[candidate_indices]
        faiss.normalize_L2(candidate_emb)
        q_emb = query_embeddings[i:i+1]
        faiss.normalize_L2(q_emb)
        index = faiss.IndexFlatIP(candidate_emb.shape[1])
        index.add(candidate_emb)
        D, I = index.search(q_emb, min(TOP_K_FINAL, len(candidates)))
        reranked[qid] = [(candidates[idx], float(D[0][rank])) for rank, idx in enumerate(I[0])]
    return reranked

def write_run_file(reranked_results, output_file):
    with open(output_file, "w") as f:
        for qid, docs in reranked_results.items():
            for rank, (pid, score) in enumerate(docs, 1):
                f.write(f"{qid} Q0 {pid} {rank} {score} hybrid_hnsw\n")

for bm25_file in BM25_FILES:
    print(f"Processing {bm25_file}")
    bm25_results = load_bm25_results(bm25_file)
    reranked_results = hybrid_rerank(query_embeddings, query_ids, passage_embeddings, passage_ids, bm25_results)
    out_file = bm25_file.replace(".txt", "_hybrid_hnsw.txt")
    write_run_file(reranked_results, out_file)
    print(f"Written: {out_file}")
