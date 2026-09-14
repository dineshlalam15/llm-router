import os
import yaml
from typing import Dict, Any
from router_engine.models.embeddings import Embedder
from router_engine.models.svm_router import SVMRouter
from router_engine.models.knn_router import KNNRouter

class DynamicRouter:
    def __init__(self, config_path: str = "configs/router.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 1. Initialize sentence transformer embedding model
        model_name = self.config.get("embedding", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = Embedder(model_name=model_name)

        # 2. Instantiate routers
        self.svm = SVMRouter(kernel=self.config.get("training", {}).get("svm_kernel", "rbf"))
        self.knn = KNNRouter(
            n_neighbors=self.config.get("training", {}).get("knn_neighbors", 5),
            weights=self.config.get("training", {}).get("knn_weights", "distance"),
            metric=self.config.get("training", {}).get("knn_metric", "cosine")
        )

        # 3. Load trained model weights from disk
        model_dir = self.config.get("persistence", {}).get("model_dir", "models_out/")
        svm_path = os.path.join(model_dir, "svm.joblib")
        knn_path = os.path.join(model_dir, "knn.joblib")

        self.svm.load(svm_path)
        self.knn.load(knn_path)

    def route(self, query: str) -> Dict[str, Any]:
        """
        Executes the hierarchical routing pipeline:
        Query -> Dense Vector -> SVM -> (if low confidence) -> KNN -> (if low confidence) -> Default Gateway
        """
        query_clean = query.strip()
        threshold = float(self.config.get("routing", {}).get("confidence_threshold", 0.65))
        default_provider = self.config.get("routing", {}).get("default_provider", "litellm")

        # Step 1: Compute dense embedding
        query_emb = self.embedder.encode([query_clean])[0]

        # Step 2: Primary Prediction via SVM
        svm_provider, svm_conf, svm_probs = self.svm.predict([query_emb])

        if svm_conf >= threshold:
            return {
                "query": query,
                "provider": svm_provider,
                "confidence": round(svm_conf, 4),
                "method": "svm_primary",
                "probabilities": svm_probs,
                "fallback_triggered": False
            }

        # Step 3: Fallback 1 -> KNN Similarity Routing
        knn_provider, knn_conf, knn_probs = self.knn.predict(query_emb)

        if knn_conf >= threshold:
            return {
                "query": query,
                "provider": knn_provider,
                "confidence": round(knn_conf, 4),
                "method": "knn_fallback",
                "probabilities": knn_probs,
                "fallback_triggered": True
            }

        # Step 4: Fallback 2 -> Default Master Proxy / Gateway
        return {
            "query": query,
            "provider": default_provider,
            "confidence": round(svm_conf, 4),
            "method": "default_gateway_fallback",
            "probabilities": svm_probs,
            "fallback_triggered": True
        }