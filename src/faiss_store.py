import json
from pathlib import Path
import faiss
import numpy as np
from .config import FAISS_INDEX_PATH, METADATA_PATH, VECTOR_DIR

class FaissStore:
    def __init__(self, index=None, metadata=None):
        self.index = index
        self.metadata = metadata or []

    def build(self, embeddings: np.ndarray, metadata: list[dict]):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        # Embeddings are normalized, so inner product behaves like cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = metadata

    def save(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        Path(metadata_path).write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        if not Path(index_path).exists() or not Path(metadata_path).exists():
            raise FileNotFoundError("FAISS index not found. Run: python build_index.py")

        index = faiss.read_index(str(index_path))
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(index=index, metadata=metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 4):
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
