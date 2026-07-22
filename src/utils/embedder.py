from __future__ import annotations
from sentence_transformers import SentenceTransformer, CrossEncoder
import config

_model: SentenceTransformer | None = None
_nli_model: CrossEncoder | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(config.NLI_MODEL)
    return _nli_model
