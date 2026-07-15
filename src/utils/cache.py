from __future__ import annotations
import sqlite3
import hashlib
import json
from pathlib import Path


class ExtractionCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        finally:
            conn.close()

    def get(self, key: str) -> dict | None:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()

    def set(self, key: str, value: dict) -> None:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
        finally:
            conn.close()

    @staticmethod
    def make_key(content: str, prefix: str = "") -> str:
        digest = hashlib.sha256(content.encode()).hexdigest()
        return f"{prefix}:{digest}"
