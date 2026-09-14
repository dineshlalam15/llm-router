import os
import yaml
from llm_router.models.embeddings import Embedder
from llm_router.models.svm_router import SVMRouter
from llm_router.models.knn_router import KNNRouter

class DynamicRouter:
    def __init__(self, config_path="configs/router.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.embedder = Embedder(self.config['embedding']['model_name'])
        self.svm = SVMRouter()
        self.knn = KNNRouter()
        
        # Load persisted models
        model_dir = self.config['persistence']['model_dir']
        self.svm.load(os.path.join(model_dir, "svm.joblib"))
        self.knn.load(os.path.join(model_dir, "knn.joblib"))
        
    def route(self, query: str) -> dict:
        threshold = self.config['routing']['confidence_threshold']
        
        # 1. Embed Query
        emb = self.embedder.encode([query])
        
        # 2. Primary Prediction (SVM)
        provider, conf, probs = self.svm.predict(emb)
        method = "svm"
        
        # 3. Secondary Fallback (KNN)
        if conf < threshold:
            knn_provider, knn_conf, knn_probs = self.knn.predict(emb)
            if knn_conf >= threshold:
                provider, conf, probs = knn_provider, knn_conf, knn_probs
                method = "knn_fallback"
            else:
                # 4. Final Safety Fallback
                provider = self.config['routing']['default_provider']
                method = "default_fallback"
                
        return {
            "query": query,
            "provider": provider,
            "confidence": conf,
            "method": method,
            "probabilities": probs
        }