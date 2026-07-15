import pytest
from pathlib import Path
from src.utils.pdf_extractor import extract_pdf, quality_check


def test_quality_check_passes_on_good_text():
    text = "John Smith\nSoftware Engineer\n" + ("Python Django REST API " * 20)
    assert quality_check(text) is True


def test_quality_check_fails_on_short_text():
    assert quality_check("Hi") is False


def test_quality_check_fails_on_empty():
    assert quality_check("") is False


def test_extract_pdf_returns_dict_with_required_keys():
    pdf_path = Path("resume/cv_00009.pdf")
    if not pdf_path.exists():
        pytest.skip("CV file not present")
    result = extract_pdf(str(pdf_path))
    assert "raw_text" in result
    assert "extraction_status" in result
    assert "source_file" in result
    assert result["extraction_status"] in ("ok", "low_confidence")


def test_extract_pdf_ok_status_for_real_cv():
    pdf_path = Path("resume/cv_00009.pdf")
    if not pdf_path.exists():
        pytest.skip("CV file not present")
    result = extract_pdf(str(pdf_path))
    assert result["extraction_status"] == "ok"
    assert len(result["raw_text"]) > 200
