from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class HQFDEConfig(BaseSettings):
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path(__file__).parent.parent / "indexer" / "data")
    output_dir: Path = Field(default=Path(__file__).parent / "output")
    cache_dir: Path = Field(default=Path(__file__).parent / "cache")

    input_tsv: str = Field(default="collection.tsv")
    output_tsv: str = Field(default="expanded_passages.tsv")

    llm_model_name: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct")
    llm_max_new_tokens: int = Field(default=256)
    llm_temperature: float = Field(default=0.7)

    doc2query_model: str = Field(default="castorini/doc2query-t5-base-msmarco")
    num_queries_per_doc: int = Field(default=5)

    nli_model: str = Field(default="microsoft/deberta-v3-large-mnli")
    nli_entailment_threshold: float = Field(default=0.9)

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dedup_similarity_threshold: float = Field(default=0.85)

    batch_size: int = Field(default=8)
    max_doc_length: int = Field(default=512)
    device: str = Field(default="mps")

    indexer_api_url: str = Field(default="http://localhost:8080")

    class Config:
        env_prefix = "HQFDE_"


config = HQFDEConfig()
