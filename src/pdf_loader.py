from pathlib import Path
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n--- Page {page_no} ---\n{text}")

    return "\n".join(pages)
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += "\n" + para
        else:
            if current.strip():
                chunks.append(current.strip())

            # overlap using last part of previous chunk
            overlap_text = current[-overlap:] if current else ""
            current = overlap_text + "\n" + para

    if current.strip():
        chunks.append(current.strip())

    return chunks