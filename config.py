from __future__ import annotations
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:7b")
LARGE_MODEL = os.getenv("LARGE_MODEL", "qwen3:14b")

EXTRACTION_TEMPERATURE = float(os.getenv("EXTRACTION_TEMPERATURE", "0.0"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.1"))
JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "2"))

SKILL_MATCH_THRESHOLD = float(os.getenv("SKILL_MATCH_THRESHOLD", "0.80"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
TOP_N_FOR_CALIBRATION = int(os.getenv("TOP_N_FOR_CALIBRATION", "20"))

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
JD_DIR = BASE_DIR / "jd"
CACHE_DB_PATH = BASE_DIR / ".cache" / "extractions.db"


def ensure_dirs() -> None:
    for _d in [RESULTS_DIR, JD_DIR, CACHE_DB_PATH.parent]:
        _d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
