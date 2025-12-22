# HQF-DE: Hybrid Query- and Fact-Guided Document Expansion

A document expansion framework for improving information retrieval using LLM-based semantic expansion, NLI validation, and Doc2Query.

## Project Structure

```
wse-final-project/
├── hqf_de/                     # Main HQF-DE implementation
│   ├── models/                 # ML model wrappers
│   │   ├── llm.py              # Llama-3 semantic expander
│   │   ├── doc2query.py        # Doc2Query T5 model
│   │   ├── nli.py              # DeBERTa NLI validator
│   │   └── embeddings.py       # Sentence embeddings
│   ├── pipeline/               # Pipeline components
│   │   ├── expander.py         # Main HQF-DE pipeline
│   │   ├── combiner.py         # Expansion combiner
│   │   └── indexer_bridge.py   # C++ indexer integration
│   ├── evaluation/             # Evaluation tools
│   │   ├── metrics.py          # nDCG, Recall, MRR, MAP
│   │   └── evaluator.py        # Evaluation pipeline
│   ├── hqf_de_colab.ipynb      # Colab notebook (GPU)
│   ├── run_pipeline.py         # CLI runner
│   └── README.md               # Detailed documentation
│
├── indexer/                    # C++ BM25 search indexer
│   ├── indexer.cpp             # Build inverted index
│   ├── merger.cpp              # Merge partial indexes
│   ├── query.cpp               # Query processing & server
│   └── Crow/                   # HTTP server library
│
├── data/                       # Input data
│   ├── collection_100k.tsv     # 100K MS MARCO documents
│   ├── qrels.dev.tsv           # Relevance judgments
│   └── *.h5                    # Embeddings
│
├── output/                     # Generated outputs
│   └── expanded_100k.tsv       # Expanded documents
│
├── shortcuts/                  # API-based shortcuts (for speed)
│   ├── hqf_de.go               # Go implementation (OpenRouter API)
│   └── hqf_de_api.py           # Python API version
│
└── *.pdf                       # Reference papers
```

## Pipeline Overview

The HQF-DE pipeline consists of three phases:

### Phase 1: LLM Semantic Expansion
- Uses **Llama-3-8B** to analyze documents
- Identifies semantic gaps (temporal, entity, contextual)
- Generates factual expansions to fill gaps

### Phase 2: NLI Validation
- Uses **DeBERTa-v3-large** for entailment checking
- Filters out hallucinated/incorrect expansions
- Only keeps expansions with entailment score > 0.9

### Phase 3: Hybrid Integration
- Generates synthetic queries using **Doc2Query T5**
- Combines validated expansions + synthetic queries
- Deduplicates using **Sentence-BERT** embeddings
- Selects diverse expansions using MMR

## Quick Start

### Option 1: Full Pipeline (GPU Required)

```bash
cd hqf_de

# Install dependencies
pip install -r requirements.txt

# Run on sample document
python run_pipeline.py --demo "The Amazon rainforest is the world's largest tropical rainforest."

# Expand documents
python run_pipeline.py --expand --limit 1000
```

### Option 2: Colab Notebook (Free GPU)

Open `hqf_de/hqf_de_colab.ipynb` in Google Colab for GPU acceleration.

### Option 3: API Shortcut (Fastest)

For large-scale processing without local GPU:

```bash
cd shortcuts

# Build Go version
go build -o hqf_de_expand hqf_de.go

# Run expansion via OpenRouter API
./hqf_de_expand -input ../data/collection_100k.tsv \
                -output ../output/expanded.tsv \
                -workers 50
```

## Indexing & Evaluation

```bash
cd indexer

# Build index from expanded documents
./indexer ../output/expanded_100k.tsv

# Run search server
./query --server

# Evaluate with qrels
# TODO: Add evaluation script
```

## Results

| Method | nDCG@10 | Recall@1000 |
|--------|---------|-------------|
| BM25 Baseline | TBD | TBD |
| + Doc2Query | TBD | TBD |
| + HQF-DE | TBD | TBD |

## References

1. Nogueira et al. "Document Expansion by Query Prediction" (2019)
2. Gao et al. "Precise Zero-Shot Dense Retrieval" (2022)
3. He et al. "DeBERTa: Decoding-enhanced BERT" (2021)

## License

MIT
