from unittest.mock import MagicMock, patch

import pytest

from agentic_bi.llm.multimodal_client import get_multimodal_llm


@pytest.fixture(autouse=True)
def clear_llm_cache():
    get_multimodal_llm.cache_clear()
    yield
    get_multimodal_llm.cache_clear()


def make_settings(provider: str, model: str):
    settings = MagicMock()
    settings.multimodal_llm_provider = provider
    settings.multimodal_llm_model = model
    settings.llm_temperature = 0.0
    settings.openai_api_key = "test-openai-key"
    settings.ollama_api_key = "test-ollama-key"
    settings.ollama_cloud_base_url = "https://ollama.com/v1"
    return settings


@patch("agentic_bi.llm.multimodal_client.get_settings")
@patch("langchain_openai.ChatOpenAI")
def test_get_multimodal_llm_openai(mock_chat_openai, mock_get_settings):
    mock_get_settings.return_value = make_settings("openai", "gpt-5-nano")

    get_multimodal_llm()

    mock_chat_openai.assert_called_once_with(
        model="gpt-5-nano",
        temperature=0.0,
        api_key="test-openai-key",
    )


@patch("agentic_bi.llm.multimodal_client.get_settings")
@patch("langchain_openai.ChatOpenAI")
def test_get_multimodal_llm_ollama(mock_chat_openai, mock_get_settings):
    mock_get_settings.return_value = make_settings("ollama", "gemma4:31b")

    get_multimodal_llm()

    mock_chat_openai.assert_called_once_with(
        model="gemma4:31b",
        temperature=0.0,
        api_key="test-ollama-key",
        base_url="https://ollama.com/v1",
    )


@patch("agentic_bi.llm.multimodal_client.get_settings")
@patch("langchain_openai.ChatOpenAI")
def test_temperature_override(mock_chat_openai, mock_get_settings):
    mock_get_settings.return_value = make_settings("openai", "gpt-5-nano")

    get_multimodal_llm(temperature=0.7)

    mock_chat_openai.assert_called_once_with(
        model="gpt-5-nano",
        temperature=0.7,
        api_key="test-openai-key",
    )


@patch("agentic_bi.llm.multimodal_client.get_settings")
def test_unsupported_provider_raises_error(mock_get_settings):
    mock_get_settings.return_value = make_settings("anthropic", "test-model")

    with pytest.raises(ValueError, match="Unsupported multimodal llm_provider"):
        get_multimodal_llm()