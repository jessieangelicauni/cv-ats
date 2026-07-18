from __future__ import annotations
from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_ollama import ChatOllama

T = TypeVar("T", bound=BaseModel)


def get_llm(model: str, schema: Type[T], temperature: float = 0.0):
    """Return a structured-output LLM chain that returns validated instances of `schema`."""
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm.with_structured_output(schema)
