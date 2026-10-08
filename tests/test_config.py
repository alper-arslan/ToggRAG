from pathlib import Path

import pytest
from pydantic import ValidationError

from toggrag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    RerankerConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
    load_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestChunkingConfig:
    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.target_tokens == 400

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValidationError):
            ChunkingConfig(overlap_ration=0.2)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_target_tokens_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            ChunkingConfig(target_tokens=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 5])
    def test_overlap_ratio_must_be_in_range(self, bad):
        with pytest.raises(ValidationError):
            ChunkingConfig(overlap_ratio=bad)

    def test_zero_overlap_is_allowed(self):
        assert ChunkingConfig(overlap_ratio=0).overlap_ratio == 0


class TestRerankerConfig:
    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValidationError):
            RerankerConfig(provider="nonsense")

    def test_top_n_cannot_exceed_candidates(self):
        with pytest.raises(ValidationError, match="top_n"):
            RerankerConfig(candidates=3, top_n=5)

    @pytest.mark.parametrize("field", ["candidates", "top_n"])
    @pytest.mark.parametrize("bad", [0, -1])
    def test_counts_must_be_positive(self, field, bad):
        with pytest.raises(ValidationError):
            RerankerConfig(**{field: bad})


class TestRetrievalConfig:
    @pytest.mark.parametrize(
        "field", ["bm25_top_k", "dense_top_k", "rrf_k", "final_top_k"]
    )
    @pytest.mark.parametrize("bad", [0, -1])
    def test_values_must_be_positive(self, field, bad):
        with pytest.raises(ValidationError):
            RetrievalConfig(**{field: bad})

    def test_final_top_k_cannot_exceed_candidates(self):
        with pytest.raises(ValidationError, match="final_top_k"):
            RetrievalConfig(bm25_top_k=3, dense_top_k=4, final_top_k=5)

    def test_final_top_k_may_equal_larger_candidate_list(self):
        cfg = RetrievalConfig(bm25_top_k=3, dense_top_k=5, final_top_k=5)
        assert cfg.final_top_k == 5


class TestEmbeddingConfig:
    @pytest.mark.parametrize("provider", ["ollama", "sentence-transformers"])
    def test_known_providers_are_accepted(self, provider):
        assert EmbeddingConfig(provider=provider).provider == provider

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(provider="nonsense")


class TestLLMConfig:
    @pytest.mark.parametrize("provider", ["ollama", "openai", "anthropic"])
    def test_known_providers_are_accepted(self, provider):
        assert LLMConfig(provider=provider).provider == provider

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider="nonsense")

    @pytest.mark.parametrize("bad", [-0.1, 2.1])
    def test_temperature_must_be_in_range(self, bad):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_max_tokens_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=bad)


class TestVectorStoreConfig:
    def test_unknown_distance_is_rejected(self):
        with pytest.raises(ValidationError):
            VectorStoreConfig(distance="manhattan")


class TestSettings:
    def test_real_config_yaml_loads(self, monkeypatch):
        # config.yaml is resolved relative to the working directory
        monkeypatch.chdir(REPO_ROOT)
        settings = Settings()
        assert settings.chunking.target_tokens == 400
        assert settings.vector_store.collection_name == "togg_manual"

    def test_env_var_overrides_yaml_and_keeps_siblings(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        monkeypatch.setenv("LLM__MODEL", "other-model")
        settings = Settings()
        assert settings.llm.model == "other-model"
        assert settings.llm.base_url == "http://localhost:11434/v1"

    def test_load_settings_reads_secret_from_dotenv(self, monkeypatch, tmp_path):
        monkeypatch.chdir(REPO_ROOT)
        # load_dotenv writes into os.environ. delenv alone records nothing when the
        # variable is absent, so set it first: monkeypatch then removes it afterwards.
        for name in ("ANTHROPIC_API_KEY", "UNRELATED"):
            monkeypatch.setenv(name, "placeholder")
            monkeypatch.delenv(name)
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("ANTHROPIC_API_KEY=from-dotenv\nUNRELATED=ignored\n")

        settings = load_settings(dotenv_file)

        assert settings.anthropic_api_key.get_secret_value() == "from-dotenv"
        assert "from-dotenv" not in repr(settings)

    def test_real_environment_beats_dotenv(self, monkeypatch, tmp_path):
        monkeypatch.chdir(REPO_ROOT)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("ANTHROPIC_API_KEY=from-dotenv\n")

        settings = load_settings(dotenv_file)

        assert settings.anthropic_api_key.get_secret_value() == "from-shell"
