
from pathlib import Path
import pymupdf


def load_pdf(path):
    pages = []

    with pymupdf.open(path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text(sort=True).strip()

            if text:
                pages.append({
                    "page": page_num,
                    "text": text
                })

    return pages


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data" / "optimization.pdf"

    pages = load_pdf(path)

    print("Pages with text:", len(pages))
    print("Total characters:", sum(len(p["text"]) for p in pages))

    if pages:
        print("\nFirst 500 characters:")
        print(pages[0]["text"][:500])
