import json

import httpx
import pytest

from toggrag.config import Settings
from toggrag.llm import AnthropicLLM, OpenAICompatibleLLM, make_llm


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
    assert isinstance(make_llm(_settings()), OpenAICompatibleLLM)
