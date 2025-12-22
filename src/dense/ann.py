import h5py
import numpy as np
import faiss
from tqdm import tqdm

def load_h5_embeddings(file_path, id_key='id', embedding_key='embedding'):
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(str)
        embeddings = np.array(f[embedding_key]).astype(np.float32)
    return ids, embeddings

passage_ids, passage_embeddings = load_h5_embeddings("ms_marco/msmarco_passages_embeddings_subset.h5")
query_ids, query_embeddings = load_h5_embeddings("ms_marco/msmarco_queries_dev_eval_embeddings.h5")

faiss.normalize_L2(passage_embeddings)
faiss.normalize_L2(query_embeddings)

dim = passage_embeddings.shape[1]
index = faiss.IndexHNSWFlat(dim, 16)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 256
faiss.omp_set_num_threads(faiss.omp_get_max_threads())
index.add(passage_embeddings)

K = 200
batch_size = 100
num_batches = (len(query_embeddings) + batch_size - 1) // batch_size

with open("run_hnsw.txt", "w") as f:
    for start in tqdm(range(0, len(query_embeddings), batch_size), total=num_batches, desc="Retrieving"):
        batch = query_embeddings[start:start+batch_size]
        distances, labels = index.search(batch, K)
        for qi, qid in enumerate(query_ids[start:start+batch_size]):
            qid = str(qid).strip()
            for rank, (pid_idx, score) in enumerate(zip(labels[qi], distances[qi])):
                if rank >= 100:
                    break
                pid = str(passage_ids[pid_idx]).strip()
                f.write(f"{qid} Q0 {pid} {rank+1} {score} hnsw_system\n")
