# tests/test_main_flags.py
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import tempfile
import os
from main import app


runner = CliRunner()


def _make_state():
    rc = MagicMock()
    rc.candidate_id = "cand_a"
    rc.calibrated_score = 85.0
    rc.delta_from_raw = 0.0
    ranked = MagicMock()
    ranked.ranked_candidates = [rc]
    ranked.pool_summary = ""
    ranked.calibration_rationale = ""
    jd = MagicMock()
    jd.role_title = "SWE"
    state = MagicMock()
    state.final_ranking = ranked
    state.jd_structured = jd
    state.candidate_assessments = []
    state.cv_profiles = []
    state.otel_trace_id = "trace-123"
    state.hallucination_flags = []
    return state


def _baseline_results():
    return {
        "tfidf":   {"ranking": ["cand_a"], "scores": {"cand_a": 0.9},
                    "distribution": {"mean": 0.9, "std": 0.0, "min": 0.9, "max": 0.9}},
        "keyword": {"ranking": ["cand_a"], "scores": {"cand_a": 3},
                    "distribution": {"mean": 3.0, "std": 0.0, "min": 3.0, "max": 3.0}},
        "evidencerank": {"ranking": ["cand_a"]},
        "cross_method_tau": {
            "tfidf_vs_keyword": 1.0,
            "tfidf_vs_evidencerank": 1.0,
            "keyword_vs_evidencerank": 1.0,
        },
    }


def _ablation_results():
    metrics = {"hallucination_rate": 0.0, "score_std": 0.0, "llm_calls": 2, "tau_vs_full": 1.0}
    return {
        "full_system": metrics,
        "no_rag": metrics,
        "no_evidence_grounding": metrics,
        "no_skill_filter": metrics,
        "no_calibration": metrics,
    }


def _run_with_flags(tmp_path: Path, *extra_flags):
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("Senior SWE role")
    cv_dir = tmp_path / "cvs"
    cv_dir.mkdir()

    state = _make_state()
    mock_pdf = {"candidate_id": "cand_a", "raw_text": "Python developer", "extraction_status": "ok"}

    with patch("main._load_cvs", return_value=[mock_pdf]), \
         patch("main.run_pipeline", return_value=state), \
         patch("main.generate_report"), \
         patch("main.setup_telemetry"), \
         patch("main.shutdown"), \
         patch("main.get_langfuse", return_value=MagicMock()), \
         patch("main.get_tracer", return_value=MagicMock()):
        result = runner.invoke(
            app,
            ["--jd", str(jd_file), "--cv-dir", str(cv_dir), "--output", str(tmp_path / "out"),
             *extra_flags],
        )
    return result


def test_baselines_flag_calls_run_baselines(tmp_path):
    with patch("main.run_baselines", return_value=_baseline_results()) as mock_bl, \
         patch("main._load_cvs", return_value=[{"candidate_id": "cand_a", "raw_text": "x", "extraction_status": "ok"}]), \
         patch("main.run_pipeline", return_value=_make_state()), \
         patch("main.generate_report"), \
         patch("main.setup_telemetry"), \
         patch("main.shutdown"), \
         patch("main.get_langfuse", return_value=MagicMock()), \
         patch("main.get_tracer", return_value=MagicMock()):
        jd_file = tmp_path / "jd.txt"; jd_file.write_text("role")
        (tmp_path / "cvs").mkdir()
        result = runner.invoke(app, [
            "--jd", str(jd_file), "--cv-dir", str(tmp_path / "cvs"),
            "--output", str(tmp_path / "out"), "--baselines",
        ])
    assert result.exit_code == 0, result.output
    mock_bl.assert_called_once()


def test_baselines_flag_writes_json(tmp_path):
    out_dir = tmp_path / "out"
    with patch("main.run_baselines", return_value=_baseline_results()), \
         patch("main._load_cvs", return_value=[{"candidate_id": "cand_a", "raw_text": "x", "extraction_status": "ok"}]), \
         patch("main.run_pipeline", return_value=_make_state()), \
         patch("main.generate_report"), \
         patch("main.setup_telemetry"), \
         patch("main.shutdown"), \
         patch("main.get_langfuse", return_value=MagicMock()), \
         patch("main.get_tracer", return_value=MagicMock()):
        jd_file = tmp_path / "jd.txt"; jd_file.write_text("role")
        (tmp_path / "cvs").mkdir()
        result = runner.invoke(app, [
            "--jd", str(jd_file), "--cv-dir", str(tmp_path / "cvs"),
            "--output", str(out_dir), "--baselines",
        ])
    assert result.exit_code == 0, result.output
    # Find the run directory
    run_dirs = list(out_dir.glob("run_*"))
    assert run_dirs, "No run directory created"
    baseline_json = run_dirs[0] / "evaluation" / "baselines.json"
    assert baseline_json.exists(), f"baselines.json not found at {baseline_json}"


def test_ablation_flag_calls_run_ablation(tmp_path):
    with patch("main.run_ablation", return_value=_ablation_results()) as mock_ab, \
         patch("main._load_cvs", return_value=[{"candidate_id": "cand_a", "raw_text": "x", "extraction_status": "ok"}]), \
         patch("main.run_pipeline", return_value=_make_state()), \
         patch("main.generate_report"), \
         patch("main.setup_telemetry"), \
         patch("main.shutdown"), \
         patch("main.get_langfuse", return_value=MagicMock()), \
         patch("main.get_tracer", return_value=MagicMock()):
        jd_file = tmp_path / "jd.txt"; jd_file.write_text("role")
        (tmp_path / "cvs").mkdir()
        result = runner.invoke(app, [
            "--jd", str(jd_file), "--cv-dir", str(tmp_path / "cvs"),
            "--output", str(tmp_path / "out"), "--ablation",
        ])
    assert result.exit_code == 0, result.output
    mock_ab.assert_called_once()


def test_ablation_flag_writes_json(tmp_path):
    out_dir = tmp_path / "out"
    with patch("main.run_ablation", return_value=_ablation_results()), \
         patch("main._load_cvs", return_value=[{"candidate_id": "cand_a", "raw_text": "x", "extraction_status": "ok"}]), \
         patch("main.run_pipeline", return_value=_make_state()), \
         patch("main.generate_report"), \
         patch("main.setup_telemetry"), \
         patch("main.shutdown"), \
         patch("main.get_langfuse", return_value=MagicMock()), \
         patch("main.get_tracer", return_value=MagicMock()):
        jd_file = tmp_path / "jd.txt"; jd_file.write_text("role")
        (tmp_path / "cvs").mkdir()
        result = runner.invoke(app, [
            "--jd", str(jd_file), "--cv-dir", str(tmp_path / "cvs"),
            "--output", str(out_dir), "--ablation",
        ])
    assert result.exit_code == 0, result.output
    run_dirs = list(out_dir.glob("run_*"))
    assert run_dirs
    ablation_json = run_dirs[0] / "evaluation" / "ablation.json"
    assert ablation_json.exists(), f"ablation.json not found at {ablation_json}"


def test_neither_flag_does_not_call_baselines_or_ablation(tmp_path):
    with patch("main.run_baselines") as mock_bl, \
         patch("main.run_ablation") as mock_ab, \
         patch("main._load_cvs", return_value=[{"candidate_id": "cand_a", "raw_text": "x", "extraction_status": "ok"}]), \
         patch("main.run_pipeline", return_value=_make_state()), \
         patch("main.generate_report"), \
         patch("main.setup_telemetry"), \
         patch("main.shutdown"), \
         patch("main.get_langfuse", return_value=MagicMock()), \
         patch("main.get_tracer", return_value=MagicMock()):
        jd_file = tmp_path / "jd.txt"; jd_file.write_text("role")
        (tmp_path / "cvs").mkdir()
        result = runner.invoke(app, [
            "--jd", str(jd_file), "--cv-dir", str(tmp_path / "cvs"),
            "--output", str(tmp_path / "out"),
        ])
    assert result.exit_code == 0, result.output
    mock_bl.assert_not_called()
    mock_ab.assert_not_called()
