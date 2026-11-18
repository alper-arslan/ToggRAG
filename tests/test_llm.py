import json

import httpx
import pytest

from toggrag.config import Settings
from toggrag.llm import AnthropicLLM, OllamaLLM, OpenAICompatibleLLM, make_llm


def test_ollama_sends_context_window_and_reads_reply():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["stream"] is False
        assert body["options"] == {
            "num_ctx": 8192,
            "temperature": 0.0,
            "num_predict": 100,
        }
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert body["messages"][1] == {"role": "user", "content": "soru"}
        return httpx.Response(200, json={"message": {"content": "yanıt"}})

    client = httpx.Client(
        base_url="http://ollama", transport=httpx.MockTransport(handler)
    )
    llm = OllamaLLM("m", "http://ollama", 8192, 0.0, 100, client=client)
    assert llm.complete("sys", "soru") == "yanıt"


def _ollama(handler, context_window=8192):
    client = httpx.Client(
        base_url="http://ollama", transport=httpx.MockTransport(handler)
    )
    return OllamaLLM("m", "http://ollama", context_window, 0.0, 100, client=client)


def test_ollama_error_body_names_the_model():
    llm = _ollama(lambda request: httpx.Response(200, json={"error": "no model"}))
    with pytest.raises(RuntimeError, match="'m'.*no model"):
        llm.complete("sys", "soru")


def test_ollama_http_error_carries_the_server_message():
    llm = _ollama(
        lambda request: httpx.Response(404, json={"error": "model 'm' not found"})
    )
    with pytest.raises(RuntimeError, match="HTTP 404.*not found"):
        llm.complete("sys", "soru")


def test_ollama_warns_when_the_prompt_fills_the_context_window(caplog):
    reply = {"message": {"content": "y"}, "prompt_eval_count": 1}
    llm = _ollama(lambda request: httpx.Response(200, json=reply), context_window=101)
    with caplog.at_level("WARNING"):
        llm.complete("sys", "soru")
    assert "context window" in caplog.text


def test_ollama_warns_when_the_answer_was_cut_at_max_tokens(caplog):
    reply = {"message": {"content": "y"}, "done_reason": "length"}
    llm = _ollama(lambda request: httpx.Response(200, json=reply))
    with caplog.at_level("WARNING"):
        assert llm.complete("sys", "soru") == "y"
    assert "max_tokens" in caplog.text


def test_make_llm_drops_a_trailing_v1_from_the_ollama_url():
    settings = Settings(llm={"provider": "ollama", "base_url": "http://box:11434/v1"})
    assert str(make_llm(settings)._client.base_url).startswith("http://box:11434")
    assert "/v1" not in str(make_llm(settings)._client.base_url)


def test_make_llm_passes_the_ollama_settings_through():
    settings = Settings(
        llm={
            "provider": "ollama",
            "base_url": "http://box:11434",
            "context_window": 4096,
        }
    )
    llm = make_llm(settings)
    assert llm._options["num_ctx"] == 4096
    assert str(llm._client.base_url).startswith("http://box:11434")


def test_openai_compatible_sends_messages_and_reads_reply():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert body["messages"][1] == {"role": "user", "content": "soru"}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "yanıt"}}]}
        )

    client = httpx.Client(
        base_url="http://llm/v1", transport=httpx.MockTransport(handler)
    )
    llm = OpenAICompatibleLLM("m", "http://llm/v1", 0.0, 100, client=client)
    assert llm.complete("sys", "soru") == "yanıt"


def test_anthropic_sends_system_separately_and_joins_text_parts():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/messages"
        assert body["system"] == "sys"
        assert body["messages"] == [{"role": "user", "content": "soru"}]
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ]
            },
        )

    client = httpx.Client(
        base_url="https://api.anthropic.com", transport=httpx.MockTransport(handler)
    )
    llm = AnthropicLLM("m", 0.0, 100, "key", client=client)
    assert llm.complete("sys", "soru") == "ab"


def _settings(**overrides):
    return Settings(pdf_path="x.pdf", data_dir="data", **overrides)


def test_make_llm_requires_anthropic_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_llm(_settings(llm={"provider": "anthropic", "model": "m"}))


def test_make_llm_requires_openai_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_llm(_settings(llm={"provider": "openai", "model": "m"}))


def test_make_llm_builds_ollama_without_a_key():
    assert isinstance(make_llm(_settings()), OllamaLLM)


def test_make_llm_uses_the_openai_adapter_for_openai():
    settings = _settings(llm={"provider": "openai", "model": "m"}, openai_api_key="k")
    assert isinstance(make_llm(settings), OpenAICompatibleLLM)
