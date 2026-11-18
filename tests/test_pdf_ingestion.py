
import pymupdf
import pytest

from src.pdf_ingestion import pdf_to_evidence


def make_test_pdf():
    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "The study included 250 participants. "
        "The model was evaluated using AUROC.",
    )

    data = pdf.tobytes()
    pdf.close()
    return data


def test_pdf_to_evidence():
    data = make_test_pdf()

    units = pdf_to_evidence(
        data,
        document_id="test-paper",
    )

    assert len(units) >= 1
    assert all(
        u["document_id"] == "test-paper"
        for u in units
    )
    assert units[0]["paragraph_id"] == "page1_chunk1"
    assert units[0]["record_type"] == "paragraph"
    assert units[0]["page"] == 1
    assert "250 participants" in units[0]["text"]


def test_invalid_pdf_rejected():
    with pytest.raises(ValueError):
        pdf_to_evidence(b"not a pdf", "test-paper")


def test_empty_pdf_rejected():
    with pytest.raises(ValueError):
        pdf_to_evidence(b"", "test-paper")

