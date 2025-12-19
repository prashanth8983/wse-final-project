# HQF-DE: Hybrid Query- and Fact-Guided Document Expansion

A Python implementation of the HQF-DE framework for improving information retrieval through document expansion.

## Overview

HQF-DE combines two complementary approaches to document expansion:

1. **LLM-guided Semantic Expansion**: Uses Llama-3-8B to identify semantic gaps and generate factual expansions
2. **Doc2Query**: Generates synthetic queries that users might use to find the document
3. **NLI Validation**: Uses DeBERTa-v3-large to filter out hallucinated expansions

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon, ensure you have the right PyTorch version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### Demo on a Single Document

```bash
python run_pipeline.py --demo "The Amazon rainforest is the world's largest tropical rainforest, covering over 5.5 million square kilometers."
```

### Expand Documents

```bash
# Full HQF-DE pipeline
python run_pipeline.py --expand --limit 100

# Doc2query only (baseline)
python run_pipeline.py --expand --doc2query-only --limit 100
```

### Run Evaluation

```bash
python run_pipeline.py --evaluate --queries 100 --limit 1000
```

## Project Structure

```
hqf_de/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── cli.py               # Command-line interface
├── run_pipeline.py      # Simple run script
│
├── models/              # ML model wrappers
│   ├── llm.py           # Llama-3 semantic expander
│   ├── doc2query.py     # Doc2Query T5 model
│   ├── nli.py           # DeBERTa NLI validator
│   └── embeddings.py    # Sentence embeddings
│
├── pipeline/            # Pipeline components
│   ├── expander.py      # Main HQF-DE pipeline
│   ├── combiner.py      # Expansion combiner
│   └── indexer_bridge.py # C++ indexer integration
│
└── evaluation/          # Evaluation tools
    ├── metrics.py       # nDCG, Recall, MRR, MAP
    └── evaluator.py     # Evaluation pipeline
```

## Pipeline Phases

### Phase 1: Semantic Gap Analysis

The LLM analyzes documents to identify:
- **Temporal gaps**: Missing dates, time periods
- **Entity gaps**: Missing related entities
- **Contextual gaps**: Missing background information

Then generates expansions to fill these gaps.

### Phase 2: NLI Validation

Each expansion is validated using DeBERTa-v3-large trained on MNLI:
- Only expansions with entailment score > 0.9 are kept
- This filters out hallucinations and factually incorrect content

### Phase 3: Hybrid Integration

Combines:
- Validated semantic expansions from Phase 1
- Synthetic queries from doc2query
- Deduplicates using sentence embeddings
- Selects diverse expansions using MMR

## Configuration

Edit `config.py` or set environment variables:

```python
# Models
HQFDE_LLM_MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
HQFDE_DOC2QUERY_MODEL="castorini/doc2query-t5-base-msmarco"
HQFDE_NLI_MODEL="microsoft/deberta-v3-large-mnli"

# Thresholds
HQFDE_NLI_ENTAILMENT_THRESHOLD=0.9
HQFDE_DEDUP_SIMILARITY_THRESHOLD=0.85

# Device
HQFDE_DEVICE="mps"  # or "cuda" or "cpu"
```

## Integration with C++ Indexer

The pipeline outputs TSV files compatible with the C++ BM25 indexer:

```bash
# After expansion
cd ../indexer
./indexer ../hqf_de/output/expanded_hqfde.tsv

# Start search server
./query --server
```

## Evaluation Metrics

- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **Recall@1000**: Proportion of relevant documents in top 1000
- **MRR@10**: Mean Reciprocal Rank at rank 10
- **MAP**: Mean Average Precision

## References

1. Nogueira et al. "Document Expansion by Query Prediction" (2019)
2. Reimers & Gurevych. "Sentence-BERT" (2019)
3. Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
