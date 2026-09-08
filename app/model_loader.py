import joblib
import os

class RouterModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load_models()

    def _load_models(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        model_path = os.getenv("MODEL_PATH", os.path.join(base_dir, "model", "router_model.joblib"))
        vectorizer_path = os.getenv("VECTORIZER_PATH", os.path.join(base_dir, "model", "vectorizer.joblib"))
        
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"SUCCESS: Loaded ML models from {base_dir}/model/")
        except Exception as e:
            print(f"CRITICAL WARNING: Could not load ML models. Error: {e}")

router_ml = RouterModel()