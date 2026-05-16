"""LLM factory — configure via configs/setting.py (overridable via env vars).

Supported providers:
  openai      → OpenAI API (default). Set OPENAI_API_KEY.
  openrouter  → OpenRouter.ai (OpenAI-compatible). Set OPENROUTER_API_KEY.
  vllm        → vLLM server (OpenAI-compatible). Set VLLM_BASE_URL and LLM_MODEL.
  gemini      → Google Gemini. Set GOOGLE_API_KEY. Requires: pip install langchain-google-genai
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from configs import config

_PROVIDERS = ("openai", "openrouter", "vllm", "gemini")


def get_llm() -> BaseChatModel:
    """Return a chat model configured from configs/setting.py."""
    gen = config.generation
    provider = gen.provider.lower()

    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider={provider!r}. Choose from: {_PROVIDERS}")

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=gen.model,
            base_url="https://openrouter.ai/api/v1",
            api_key=gen.openrouter_api_key,
            temperature=gen.temperature,
        )

    if provider == "vllm":
        if not gen.vllm_base_url:
            raise ValueError(
                "VLLM_BASE_URL must be set (e.g. http://gpu-host:8000/v1). "
                "Start the server: vllm serve <model> --port 8000"
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=gen.model,
            base_url=gen.vllm_base_url,
            api_key=gen.vllm_api_key,
            temperature=gen.temperature,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for Gemini. "
                "Install it with: pip install langchain-google-genai"
            ) from e
        return ChatGoogleGenerativeAI(
            model=gen.model,
            google_api_key=gen.google_api_key,
            temperature=gen.temperature,
        )

    # openai (default) — also works for any OpenAI-compatible endpoint via openai_base_url
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=gen.model,
        api_key=gen.openai_api_key,
        base_url=gen.openai_base_url,
        temperature=gen.temperature,
    )
