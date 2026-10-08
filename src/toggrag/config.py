import pathlib
from typing import Literal

import dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class StrictModel(BaseModel):
    """Base for all config sections: unknown keys raise instead of being ignored."""

    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(StrictModel):
    """How the manual is split into chunks during ingestion."""

    target_tokens: int = Field(
        default=400, gt=0, description="Approximate chunk size in tokens."
    )
    overlap_ratio: float = Field(
        default=0.12,
        ge=0,
        lt=1,
        description="Fraction of a chunk repeated at the start of the next one.",
    )


class EmbeddingConfig(StrictModel):
    """Dense embedding model used for vector search."""

    provider: Literal["ollama", "sentence-transformers"] = Field(
        default="ollama", description="Adapter that serves the embedding model."
    )
    model: str = Field(default="bge-m3", description="Embedding model name.")
    base_url: str = Field(
        default="http://localhost:11434",
        description="Server URL for HTTP providers (e.g. Ollama).",
    )


class RerankerConfig(StrictModel):
    """Optional cross-encoder that reorders fused retrieval results."""

    enabled: bool = Field(default=False, description="Turn reranking on or off.")
    provider: Literal["sentence-transformers"] = Field(
        default="sentence-transformers", description="Adapter that serves the reranker."
    )
    model: str = Field(
        default="BAAI/bge-reranker-v2-m3", description="Cross-encoder model name."
    )
    candidates: int = Field(
        default=20, gt=0, description="How many fused results are sent to the reranker."
    )
    top_n: int = Field(
        default=5, gt=0, description="How many results are kept after reranking."
    )

    @model_validator(mode="after")
    def top_n_within_candidates(self):
        if self.top_n > self.candidates:
            raise ValueError("top_n must be <= candidates")
        return self


class RetrievalConfig(StrictModel):
    """Hybrid retrieval: BM25 + dense search, fused with Reciprocal Rank Fusion."""

    bm25_top_k: int = Field(
        default=20, gt=0, description="Results taken from BM25 (lexical) search."
    )
    dense_top_k: int = Field(
        default=20, gt=0, description="Results taken from dense (vector) search."
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="RRF constant; higher values flatten the effect of rank.",
    )
    final_top_k: int = Field(
        default=5, gt=0, description="Chunks passed to the LLM as context."
    )
    query_rewrite: bool = Field(
        default=False,
        description="Rewrite the query with the LLM before retrieval (extra LLM call).",
    )

    @model_validator(mode="after")
    def final_top_k_within_candidates(self):
        if self.final_top_k > max(self.bm25_top_k, self.dense_top_k):
            raise ValueError("final_top_k must be <= max(bm25_top_k, dense_top_k)")
        return self


class LLMConfig(StrictModel):
    """Language model that writes the grounded answer."""

    provider: Literal["ollama", "openai", "anthropic"] = Field(
        default="ollama", description="Adapter that serves the LLM."
    )
    model: str = Field(default="qwen2.5:7b", description="LLM model name.")
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Server URL for HTTP providers (OpenAI-compatible endpoint).",
    )
    temperature: float = Field(
        default=0.0, ge=0, le=2, description="Sampling temperature; 0 is deterministic."
    )
    max_tokens: int = Field(
        default=1024, gt=0, description="Maximum tokens in the generated answer."
    )


class VectorStoreConfig(StrictModel):
    """Local Qdrant collection that stores chunk embeddings."""

    path: str = Field(default="data/qdrant", description="On-disk Qdrant location.")
    collection_name: str = Field(
        default="togg_manual", description="Qdrant collection name."
    )
    distance: Literal["cosine", "dot", "euclid"] = Field(
        default="cosine", description="Vector similarity metric."
    )


class Settings(BaseSettings):
    """Top-level config: config.yaml < .env < environment variables."""

    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_nested_delimiter="__",
        extra="forbid",
    )

    pdf_path: pathlib.Path
    data_dir: pathlib.Path
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,  # highest priority
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_settings(dotenv_path: pathlib.Path | None = None) -> Settings:
    """Load .env into the environment (existing variables win), then build Settings."""
    dotenv.load_dotenv(dotenv_path)
    return Settings()
