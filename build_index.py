from pathlib import Path


from src.config import DATA_DIR
from src.data_loader import load_policies, row_to_document
from src.embeddings_hf import HFEmbeddingModel
from src.faiss_store import FaissStore
from src.pdf_loader import extract_text_from_pdf, chunk_text
from src.config import DATA_DIR
from src.data_loader import load_policies, row_to_document
from src.embeddings_hf import HFEmbeddingModel
from src.faiss_store import FaissStore




PDF_DIR = DATA_DIR / "pdfs"


def load_csv_documents():
    df = load_policies()
    documents = []
    metadata = []

    for _, row in df.iterrows():
        documents.append(row_to_document(row))
        metadata.append(row.to_dict())

    return documents, metadata

def infer_product_category(file_name: str) -> str:
    name = file_name.lower()

    if "mobile" in name or "phone" in name:
        return "Mobile Phone"
    if "laptop" in name:
        return "Laptop"
    if "audio" in name or "headphone" in name or "earbud" in name:
        return "Audio Accessories"
    if "appliance" in name:
        return "Home Appliance"
    if "general" in name or "escalation" in name:
        return "General"

    return "Unknown"
def load_pdf_documents():
    documents = []
    metadata = []

    if not PDF_DIR.exists():
        return documents, metadata

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"Reading PDF: {pdf_path.name}")

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks, start=1):
            documents.append(
                f"Source File: {pdf_path.name}\n"
                f"Chunk ID: {i}\n"
                f"Policy Text: {chunk}"
            )

            metadata.append(
                {
                    "product_category": infer_product_category(pdf_path.name),
                    "policy_section": f"{pdf_path.name} | Chunk {i}",
                    "clause_type": "pdf_chunk",
                    "clause_text": chunk,
                    "required_documents": "Not specified",
                    "risk_type": "pdf",
                    "source_file": pdf_path.name,
                    "chunk_id": i,
                }
            )

    return documents, metadata


def main():
    csv_documents, csv_metadata = load_csv_documents()
    pdf_documents, pdf_metadata = load_pdf_documents()

    documents = csv_documents + pdf_documents
    metadata = csv_metadata + pdf_metadata

    print(f"CSV policy rows: {len(csv_documents)}")
    print(f"PDF chunks: {len(pdf_documents)}")
    print(f"Total documents for FAISS: {len(documents)}")

    if not documents:
        raise ValueError("No documents found. Add CSV rows or PDFs first.")

    print("Loading Hugging Face embedding model...")
    embedder = HFEmbeddingModel()

    print("Creating embeddings...")
    embeddings = embedder.encode(documents)

    print("Building FAISS index...")
    store = FaissStore()
    store.build(embeddings, metadata)
    store.save()

    print("FAISS index saved to vector_store/")


if __name__ == "__main__":
    main()