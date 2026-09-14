# llm_router/models/embeddings.py
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # all-MiniLM-L6-v2 is selected for production MVP: low latency, 384 dimensions, runs well on CPU.
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: list):
        return self.model.encode(texts)