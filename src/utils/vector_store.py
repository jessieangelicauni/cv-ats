from __future__ import annotations
import hashlib
import re
from pathlib import Path
from src.utils.embedder import get_embedder


def _chunk_text(text: str, min_len: int = 10) -> list[str]:
    chunks = re.split(r'\n\s*\n', text)
    return [c.strip() for c in chunks if len(c.strip()) >= min_len]


class CVVectorStore:
    def __init__(self, path: Path) -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection("cv_chunks")

    def index_cv(self, candidate_id: str, raw_text: str) -> None:
        cv_hash = hashlib.sha256(raw_text.encode()).hexdigest()[:12]

        existing = self._col.get(
            where={"candidate_id": candidate_id},
            limit=1,
        )
        if existing["ids"]:
            if existing["metadatas"][0].get("cv_hash") == cv_hash:
                return
            self._col.delete(where={"candidate_id": candidate_id})

        chunks = _chunk_text(raw_text)
        if not chunks:
            return

        embedder = get_embedder()
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{candidate_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"candidate_id": candidate_id, "chunk_index": i, "cv_hash": cv_hash}
            for i in range(len(chunks))
        ]
        self._col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    def close(self) -> None:
        """Release the ChromaDB client connection (important on Windows)."""
        try:
            self._client.clear_system_cache()
        except Exception:
            pass

    def retrieve(self, candidate_id: str, query: str, top_k: int = 5) -> list[str]:
        existing = self._col.get(where={"candidate_id": candidate_id})
        n_available = len(existing["ids"])
        if n_available == 0:
            return []

        embedder = get_embedder()
        query_embedding = embedder.encode([query])[0].tolist()
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, n_available),
            where={"candidate_id": candidate_id},
            include=["documents"],
        )
        return results["documents"][0] if results["documents"] else []
