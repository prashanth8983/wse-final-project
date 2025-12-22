# HQF-DE: Hybrid Query-Focused Document Expansion

Document expansion for information retrieval using LLM expansion, NLI validation, and Doc2Query.

## Structure

```
src/
├── bm25/           # C++ BM25 indexer and query
├── dense/          # HNSW retrieval and hybrid fusion
├── models/         # LLM, NLI, Doc2Query, embeddings
├── pipeline/       # Expansion pipeline
├── notebooks/      # Colab notebooks
└── evaluation/     # Metrics (nDCG, MRR, Recall)

paper/              # Research paper and appendix
```

## Pipeline

1. **LLM Expansion** - Llama-3-8B generates semantic expansions
2. **NLI Validation** - BART-MNLI filters hallucinations
3. **Doc2Query** - T5 generates synthetic queries
4. **Hybrid Retrieval** - BM25 + Dense with RRF fusion

## Usage

Run notebooks on Google Colab (A100 recommended):
- `llm_expansion.ipynb` - LLM expansion (~11 hrs)
- `nli_validation.ipynb` - NLI filtering (~1 hr)
- `doc2query.ipynb` - Query generation (~1 hr)
- `embeddings.ipynb` - Generate embeddings (~20 min)

Local evaluation:
```bash
cd src/bm25
./indexer ../../data/expanded_100k.tsv
./merger <num_runs>
./query queries.tsv
```

## Results (TREC DL 2019)

| Method | MRR@10 |
|--------|--------|
| BM25 Original | 0.744 |
| BM25 + Doc2Query | 0.844 |
| Hybrid + Doc2Query | 0.896 |

## Models

- LLM: meta-llama/Meta-Llama-3-8B-Instruct
- NLI: facebook/bart-large-mnli
- Doc2Query: castorini/doc2query-t5-base-msmarco
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
