from __future__ import annotations
from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig

T = TypeVar("T", bound=BaseModel)


def get_llm(model: str, schema: Type[T], temperature: float = 0.0):
    """Return a structured-output LLM chain that returns validated instances of `schema`."""
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm.with_structured_output(schema)


def invoke_with_telemetry(chain, messages: list, run_name: str | None = None):
    """
    Invoke a LangChain chain with a Langfuse CallbackHandler attached.

    The handler automatically inherits the current OTel span context as parent,
    so all LLM generations appear as children of whatever phase/candidate span
    is active in the calling thread. run_name labels the generation itself —
    without it every call shows up in Langfuse as the generic "RunnableSequence".
    """
    from src.utils.telemetry import make_callback
    return chain.invoke(
        messages, RunnableConfig(callbacks=[make_callback()], run_name=run_name)
    )
