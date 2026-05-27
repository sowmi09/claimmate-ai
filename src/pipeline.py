from .config import TOP_K
from .embeddings_hf import HFEmbeddingModel
from .faiss_store import FaissStore
from .claim_router import route_query
from .claim_analyzer import analyze_claim
def infer_query_product(query: str) -> str | None:
        q = query.lower()

        if any(word in q for word in ["phone", "mobile", "imei", "charging", "screen", "battery"]):
            return "Mobile Phone"

        if any(word in q for word in ["laptop", "keyboard", "motherboard", "display"]):
            return "Laptop"

        if any(word in q for word in ["headphone", "earbud", "audio", "speaker"]):
            return "Audio Accessories"

        if any(word in q for word in ["ac", "fridge", "washing machine", "appliance"]):
            return "Home Appliance"

        return None
class ClaimMatePipeline:
    def __init__(self):
        self.embedding_model = HFEmbeddingModel()
        self.store = FaissStore.load()
        
    

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        product = infer_query_product(query)

        query_embedding = self.embedding_model.encode([query])
        retrieved = self.store.search(query_embedding, top_k=top_k + 4)

        if product:
            product_results = [
                r for r in retrieved
                if r.get("product_category") == product
            ]

            if product_results:
                retrieved = product_results[:top_k]
            else:
                retrieved = retrieved[:top_k]
        else:
            retrieved = retrieved[:top_k]
        route = route_query(query)
        
        
        answer = analyze_claim(query=query, route=route, retrieved=retrieved)
        return {
            "route": route,
            "answer": answer,
            "retrieved": retrieved,
        }
    