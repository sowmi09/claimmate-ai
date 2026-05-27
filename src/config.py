from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROMPT_DIR = ROOT_DIR / "prompts"
VECTOR_DIR = ROOT_DIR / "vector_store"
OUTPUT_DIR = ROOT_DIR / "outputs"

POLICY_CSV = DATA_DIR / "policies.csv"
FAISS_INDEX_PATH = VECTOR_DIR / "claimmate.faiss"
METADATA_PATH = VECTOR_DIR / "metadata.json"

# Hugging Face embedding model.
# Alternatives: "intfloat/e5-small-v2", "thenlper/gte-small"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Ollama local model. Pull it using: ollama pull qwen2.5:3b
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

TOP_K = int(os.getenv("TOP_K", "4"))
