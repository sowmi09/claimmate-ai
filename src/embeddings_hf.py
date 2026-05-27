import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from .config import EMBEDDING_MODEL_NAME

class HFEmbeddingModel:
    """
    Hugging Face embedding model using transformers directly.
    This intentionally avoids the sentence-transformers wrapper.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy().astype("float32"))

        return np.vstack(all_embeddings)
