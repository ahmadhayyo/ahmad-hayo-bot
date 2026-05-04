"""
Provider-agnostic LLM factory.

Returns a `BaseChatModel` instance configured for the provider selected in
.env (anthropic | google). The rest of the codebase asks for a model by ROLE
(agent, summarizer) and never hardcodes a vendor.

Two roles:
  - get_agent_model()      → tool-calling, planning, execution. Hot path.
  - get_summarizer_model() → cheap+fast model used to compress history.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from config import (
    ANTHROPIC_AGENT_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_SUMMARIZER_MODEL,
    GOOGLE_AGENT_MODEL,
    GOOGLE_API_KEY,
    GOOGLE_SUMMARIZER_MODEL,
    MODEL_PROVIDER,
    assert_keys_present,
)


def _build_anthropic(model: str, temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=4096,
        timeout=120,
        api_key=ANTHROPIC_API_KEY,
    )


def _build_google(model: str, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
        # convert_system_message_to_human is needed for some Gemini variants
        # that don't accept system role; safe default.
        convert_system_message_to_human=False,
        max_output_tokens=4096,
    )


@lru_cache(maxsize=1)
def get_agent_model() -> BaseChatModel:
    """Main reasoning/tool-calling model. Cached — singleton per process."""
    assert_keys_present()
    if MODEL_PROVIDER == "anthropic":
        return _build_anthropic(ANTHROPIC_AGENT_MODEL, temperature=0.2)
    return _build_google(GOOGLE_AGENT_MODEL, temperature=0.2)


@lru_cache(maxsize=1)
def get_summarizer_model() -> BaseChatModel:
    """Cheap fast model for compressing conversation history."""
    assert_keys_present()
    if MODEL_PROVIDER == "anthropic":
        return _build_anthropic(ANTHROPIC_SUMMARIZER_MODEL, temperature=0.0)
    return _build_google(GOOGLE_SUMMARIZER_MODEL, temperature=0.0)
