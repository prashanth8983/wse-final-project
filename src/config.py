from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DOC2QUERY_MODEL = "castorini/doc2query-t5-base-msmarco"
NLI_MODEL = "facebook/bart-large-mnli"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.7
NUM_QUERIES = 5
BATCH_SIZE = 8
MAX_DOC_LENGTH = 512
NLI_THRESHOLD = 0.9
DEDUP_THRESHOLD = 0.85
DEVICE = "cuda"
