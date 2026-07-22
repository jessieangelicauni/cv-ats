from __future__ import annotations
import fitz
import pdfplumber


def quality_check(text: str) -> bool:
    has_min_length = len(text) >= 200
    has_min_words = len(text.split()) >= 50

    no_control_chars = all(c.isprintable() or c in '\n\r\t ' for c in text)

    return has_min_length and has_min_words and no_control_chars


def _extract_with_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def _extract_with_pdfplumber(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def extract_pdf(pdf_path: str) -> dict:
    text = _extract_with_pymupdf(pdf_path)

    if not quality_check(text):
        text = _extract_with_pdfplumber(pdf_path)

    status = "ok" if quality_check(text) else "low_confidence"
    return {"raw_text": text, "extraction_status": status, "source_file": pdf_path}


def raw_text_by_candidate_id(cv_raws: list[dict]) -> dict[str, str]:
    return {c["candidate_id"]: c["raw_text"] for c in cv_raws}
