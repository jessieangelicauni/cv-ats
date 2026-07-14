from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
SMALL_MODEL = "qwen2.5:7b"
LARGE_MODEL = "llama3.3:70b"

EXTRACTION_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.1

MAX_PARALLEL_WORKERS = 4
HALLUCINATION_SIMILARITY_THRESHOLD = 0.85
BORDERLINE_SCORE_THRESHOLD = 5.0
MIN_PROFILE_SKILLS = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"
JD_DIR = BASE_DIR / "jd"
CACHE_DB_PATH = BASE_DIR / ".cache" / "extractions.db"


def ensure_dirs() -> None:
    for _d in [RESULTS_DIR, GROUND_TRUTH_DIR, JD_DIR, CACHE_DB_PATH.parent]:
        _d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
