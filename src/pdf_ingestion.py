
"""Convert a PDF into TrustRAG evidence units."""

import tempfile
from pathlib import Path

from src.pdf_loader import load_pdf
from src.chunking import chunk_text


MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_EVIDENCE_UNITS = 10000


def pdf_to_evidence(
    pdf_bytes: bytes,
    document_id: str,
    chunk_size: int = 1000,
    overlap: int = 1,
):
    """Extract text and create document-scoped evidence units."""

    if not pdf_bytes:
        raise ValueError("PDF is empty")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 10 MB limit")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("File is not a valid PDF")

    # Reuse the existing path-based PDF loader.
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "uploaded.pdf"
        pdf_path.write_bytes(pdf_bytes)

        try:
            pages = load_pdf(pdf_path)
        except Exception as exc:
            raise ValueError(
                "Could not read PDF"
            ) from exc

    if not pages:
        raise ValueError(
            "No extractable text found in PDF"
        )

    units = []

    for page in pages:
        page_number = int(page["page"])

        chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, text in enumerate(chunks, start=1):
            if not text.strip():
                continue

            units.append({
                "document_id": document_id,
                "paragraph_id": f"page{page_number}_chunk{index}",
                "record_type": "paragraph",
                "page": page_number,
                "text": text,
            })

            if len(units) > MAX_EVIDENCE_UNITS:
                raise ValueError("Too many evidence units")

    if not units:
        raise ValueError("No usable text chunks")

    return units
