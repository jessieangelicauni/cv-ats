import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock


def _mock_embedder():
    rng = np.random.RandomState(42)
    mock = MagicMock()
    def encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return rng.rand(n, 384).astype(np.float32)
    mock.encode.side_effect = encode
    return mock


def test_index_cv_adds_chunks():
    from src.utils.vector_store import CVVectorStore
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with patch("src.utils.vector_store.get_embedder", return_value=_mock_embedder()):
            store = CVVectorStore(Path(tmp))
            store.index_cv("cv_001", "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
            chunks = store.retrieve("cv_001", "work experience")
            store.close()
            assert isinstance(chunks, list)
            assert len(chunks) > 0


def test_index_cv_skips_reindex_when_hash_unchanged():
    from src.utils.vector_store import CVVectorStore
    mock_emb = _mock_embedder()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with patch("src.utils.vector_store.get_embedder", return_value=mock_emb):
            store = CVVectorStore(Path(tmp))
            text = "First paragraph.\n\nSecond paragraph."
            store.index_cv("cv_001", text)
            call_count_after_first = mock_emb.encode.call_count
            store.index_cv("cv_001", text)  # same text, same hash
            store.close()
            assert mock_emb.encode.call_count == call_count_after_first


def test_index_cv_reindexes_when_content_changes():
    from src.utils.vector_store import CVVectorStore
    mock_emb = _mock_embedder()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with patch("src.utils.vector_store.get_embedder", return_value=mock_emb):
            store = CVVectorStore(Path(tmp))
            store.index_cv("cv_001", "Original content.\n\nMore original content.")
            call_count_after_first = mock_emb.encode.call_count
            store.index_cv("cv_001", "Completely different content.\n\nNew paragraph.")
            store.close()
            assert mock_emb.encode.call_count > call_count_after_first


def test_retrieve_returns_empty_for_unknown_candidate():
    from src.utils.vector_store import CVVectorStore
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with patch("src.utils.vector_store.get_embedder", return_value=_mock_embedder()):
            store = CVVectorStore(Path(tmp))
            result = store.retrieve("unknown_cv", "query")
            store.close()
            assert result == []
