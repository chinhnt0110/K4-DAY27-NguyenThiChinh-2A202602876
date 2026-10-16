"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openrouter_key and not openrouter_key.strip().endswith("..."):
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=openrouter_key,
            temperature=temperature,
        )

    if openai_key:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        if model.startswith("openai/"):
            model = model.replace("openai/", "", 1)
        base_url = os.environ.get("LLM_BASE_URL")
        kwargs = {
            "model": model,
            "api_key": openai_key,
            "temperature": temperature,
        }
        if base_url and "openrouter.ai" not in base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    raise RuntimeError("Neither a valid OPENROUTER_API_KEY nor OPENAI_API_KEY was found in .env")

