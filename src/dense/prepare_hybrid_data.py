#!/usr/bin/env python3
import os
import struct
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
VARIANTS = ["original", "expanded", "validated", "doc2query"]

def convert_embeddings_to_binary(variant):
    h5_file = f"{DATA_DIR}/embeddings_{variant}.h5"
    bin_file = f"{DATA_DIR}/embeddings_{variant}.bin"
    pids_file = f"{DATA_DIR}/passage_ids_{variant}.txt"

    if not os.path.exists(h5_file):
        print(f"  {h5_file} not found, skipping")
        return False

    with h5py.File(h5_file, 'r') as f:
        embeddings = f['embedding'][:]
        passage_ids = f['id'][:]

    num_docs, dim = len(embeddings), embeddings.shape[1]
    print(f"{variant}: {num_docs} docs, dim={dim}")

    with open(bin_file, 'wb') as f:
        f.write(struct.pack('i', num_docs))
        for emb in embeddings:
            f.write(struct.pack(f'{dim}f', *emb.astype(np.float32)))

    with open(pids_file, 'w') as f:
        for pid in passage_ids:
            f.write(f"{pid.decode('utf-8') if isinstance(pid, bytes) else pid}\n")

    print(f"  -> {bin_file} ({os.path.getsize(bin_file) / 1024 / 1024:.1f} MB)")
    return True

def generate_query_embeddings():
    print("\nGenerating query embeddings...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    queries = []
    for qf in ["Dense-Retrieval-based-Search-Engine/queries/queries.eval.tsv", "Dense-Retrieval-based-Search-Engine/queries/queries.dev.tsv"]:
        if os.path.exists(qf):
            with open(qf, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        queries.append((parts[0], parts[1]))

    print(f"  {len(queries)} queries")
    query_ids = [q[0] for q in queries]
    embeddings = model.encode([q[1] for q in queries], show_progress_bar=True, normalize_embeddings=True)

    bin_file = f"{DATA_DIR}/query_embeddings.bin"
    dim = embeddings.shape[1]
    with open(bin_file, 'wb') as f:
        f.write(struct.pack('i', len(embeddings)))
        for emb in embeddings:
            f.write(struct.pack(f'{dim}f', *emb.astype(np.float32)))

    with open(f"{DATA_DIR}/query_ids.txt", 'w') as f:
        for qid in query_ids:
            f.write(f"{qid}\n")

    print(f"  -> {bin_file} ({os.path.getsize(bin_file) / 1024 / 1024:.1f} MB)")

def main():
    print("=" * 60)
    print("Preparing data for hybrid query processor")
    print("=" * 60)

    for variant in VARIANTS:
        convert_embeddings_to_binary(variant)
    generate_query_embeddings()

    print("\n" + "=" * 60)
    print("Done!")

if __name__ == "__main__":
    main()
