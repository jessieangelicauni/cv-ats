from __future__ import annotations
from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_ollama import ChatOllama
import config

T = TypeVar("T", bound=BaseModel)


def get_llm(model: str, schema: Type[T], temperature: float = 0.0):
    """Return a structured-output LLM chain that returns validated instances of `schema`."""
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        base_url=config.OLLAMA_BASE_URL,
    )
    return llm.with_structured_output(schema)
