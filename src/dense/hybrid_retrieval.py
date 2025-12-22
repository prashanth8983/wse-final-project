#!/usr/bin/env python3
import os
import re
import argparse
import subprocess
from collections import defaultdict
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    import faiss
    import h5py
    from sentence_transformers import SentenceTransformer
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

try:
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    stemmer = PorterStemmer()
    STOPWORDS = set(stopwords.words('english'))
except ImportError:
    stemmer = None
    STOPWORDS = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do", "does", "did", "will", "would", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "what", "which", "who", "when", "where", "why", "how", "all", "each", "both", "few", "more", "most", "other", "some", "no", "not", "only", "same", "so", "than", "too", "very", "just"}

DATA_DIR = "data"
RESULTS_DIR = "results"
QUERIES_DIR = "Dense-Retrieval-based-Search-Engine/queries"
RRF_K = 60
TOP_K = 1000

VARIANTS = ["original", "expanded", "validated", "doc2query"]
VARIANT_FILES = {"original": "collection_100k.tsv", "expanded": "expanded_100k.tsv", "validated": "validated_100k.tsv", "doc2query": "doc2query_100k.tsv"}
QRELS = {"trec2019": "data/qrels.eval.one.tsv", "trec2020": "data/qrels.eval.two.tsv", "dev": "data/qrels.dev.trec.tsv"}

def tokenize(text):
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    return [stemmer.stem(t) for t in tokens] if stemmer else tokens

def load_documents(variant):
    doc_ids, doc_texts = [], []
    with open(f"{DATA_DIR}/{VARIANT_FILES[variant]}", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                doc_ids.append(parts[0])
                doc_texts.append(parts[1])
    print(f"Loaded {len(doc_ids)} documents")
    return doc_ids, doc_texts

def load_queries():
    queries = []
    for qfile in ["queries.eval.tsv", "queries.dev.tsv"]:
        filepath = f"{QUERIES_DIR}/{qfile}"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        queries.append((parts[0], parts[1]))
    return queries

def load_embeddings(variant):
    with h5py.File(f"{DATA_DIR}/embeddings_{variant}.h5", 'r') as f:
        embeddings = f['embedding'][:]
        passage_ids = [pid.decode('utf-8') if isinstance(pid, bytes) else str(pid) for pid in f['id'][:]]
    return embeddings.astype(np.float32), passage_ids

class BM25Retriever:
    def __init__(self, doc_ids, doc_texts):
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi([tokenize(text) for text in doc_texts])

    def search(self, query, top_k=TOP_K):
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

class DenseRetriever:
    def __init__(self, embeddings, passage_ids):
        self.passage_ids = passage_ids
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(embeddings)
        self.index.hnsw.efSearch = 256
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def search(self, query, top_k=TOP_K):
        query_emb = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)
        return [(self.passage_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

def reciprocal_rank_fusion(bm25_results, dense_results, k=RRF_K):
    scores = defaultdict(float)
    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        scores[doc_id] += 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(dense_results, 1):
        scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]

def run_trec_eval(qrels_file, run_file):
    try:
        result = subprocess.run(["./trec_eval_bin", "-m", "recip_rank.10", "-m", "recall.100", "-m", "ndcg_cut.10", qrels_file, run_file], capture_output=True, text=True)
        metrics = {}
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                metrics[parts[0]] = float(parts[2])
        return metrics
    except:
        return {}

def write_run_file(results, output_file, run_name):
    with open(output_file, 'w') as f:
        for qid in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x):
            for rank, (doc_id, score) in enumerate(results[qid], 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="original")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"{'='*70}\nHYBRID RETRIEVAL: {args.variant.upper()}\n{'='*70}")

    run_file = f"{RESULTS_DIR}/run_hybrid_{args.variant}.txt"

    if not args.eval_only:
        if not HAS_BM25 or not HAS_DEPS:
            print("Missing dependencies")
            return

        doc_ids, doc_texts = load_documents(args.variant)
        embeddings, passage_ids = load_embeddings(args.variant)
        queries = load_queries()

        bm25 = BM25Retriever(doc_ids, doc_texts)
        dense = DenseRetriever(embeddings, passage_ids)

        all_results = {}
        for i, (qid, query_text) in enumerate(queries):
            fused = reciprocal_rank_fusion(bm25.search(query_text), dense.search(query_text))
            all_results[qid] = fused
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(queries)} queries")

        write_run_file(all_results, run_file, f"hybrid_{args.variant}")

    print(f"\n{'='*70}\nEVALUATION\n{'='*70}")
    for qrels_name, qrels_file in QRELS.items():
        metrics = run_trec_eval(qrels_file, run_file)
        print(f"{qrels_name}: MRR={metrics.get('recip_rank_10', 0):.4f}")

if __name__ == "__main__":
    main()
