# tests/test_ablation.py
from unittest.mock import patch, MagicMock
from src.evaluation.ablation import run_ablation, ABLATION_VARIANTS


def _make_ranked(ids: list[str]):
    return [
        MagicMock(candidate_id=cid, calibrated_score=90.0 - i * 5)
        for i, cid in enumerate(ids)
    ]


def _make_state(ids: list[str]):
    state = MagicMock()
    state.final_ranking.ranked_candidates = _make_ranked(ids)
    state.candidate_assessments = [MagicMock(candidate_id=cid) for cid in ids]
    return state


def test_ablation_variants_list_has_four_entries():
    assert len(ABLATION_VARIANTS) == 4


def test_ablation_variant_names():
    names = {v["name"] for v in ABLATION_VARIANTS}
    assert names == {"no_rag", "no_evidence_grounding", "no_skill_filter", "no_calibration"}


def test_run_ablation_calls_pipeline_once_per_variant():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline",
               return_value=_make_state(["a", "b"])) as mock_pipe, \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0), \
         patch("src.evaluation.ablation.reset_call_count"), \
         patch("src.evaluation.ablation.get_call_count", return_value=2):
        run_ablation("jd", [], full_state, cv_text_map)

    assert mock_pipe.call_count == len(ABLATION_VARIANTS)


def test_run_ablation_passes_correct_kwargs_to_pipeline():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}
    captured_kwargs = []

    def capture(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return _make_state(["a", "b"])

    with patch("src.evaluation.ablation.run_pipeline", side_effect=capture), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0), \
         patch("src.evaluation.ablation.reset_call_count"), \
         patch("src.evaluation.ablation.get_call_count", return_value=2):
        run_ablation("jd", [], full_state, cv_text_map)

    no_rag_call = next(kw for kw in captured_kwargs if kw.get("use_vector_store") is False)
    assert no_rag_call is not None

    no_eg_call = next(kw for kw in captured_kwargs if kw.get("use_evidence_grounding") is False)
    assert no_eg_call is not None


def test_run_ablation_result_contains_all_variants():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline",
               return_value=_make_state(["a", "b"])), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0), \
         patch("src.evaluation.ablation.reset_call_count"), \
         patch("src.evaluation.ablation.get_call_count", return_value=2):
        result = run_ablation("jd", [], full_state, cv_text_map)

    assert "full_system" in result
    for v in ABLATION_VARIANTS:
        assert v["name"] in result


def test_run_ablation_metrics_keys():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline",
               return_value=_make_state(["a", "b"])), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0), \
         patch("src.evaluation.ablation.reset_call_count"), \
         patch("src.evaluation.ablation.get_call_count", return_value=2):
        result = run_ablation("jd", [], full_state, cv_text_map)

    for metrics in result.values():
        assert set(metrics.keys()) == {
            "hallucination_rate", "score_std", "llm_calls", "tau_vs_full"
        }


def test_full_system_tau_is_one():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline",
               return_value=_make_state(["a", "b"])), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0), \
         patch("src.evaluation.ablation.reset_call_count"), \
         patch("src.evaluation.ablation.get_call_count", return_value=2):
        result = run_ablation("jd", [], full_state, cv_text_map)

    assert result["full_system"]["tau_vs_full"] == 1.0


def test_run_ablation_resets_call_count_per_variant():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline", return_value=_make_state(["a", "b"])), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0), \
         patch("src.evaluation.ablation.reset_call_count") as mock_reset, \
         patch("src.evaluation.ablation.get_call_count", return_value=3):
        run_ablation("jd", [], full_state, cv_text_map)

    assert mock_reset.call_count == len(ABLATION_VARIANTS)
