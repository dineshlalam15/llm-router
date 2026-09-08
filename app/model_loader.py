import joblib
import os

class RouterModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load_moddel()

    def _load_model(self):
        model_path = os.getenv("MODEL_PATH", "model/router_model.joblib")
        vectorizer_path = os.getenv("VECTORIZER_PATH", "model/vectorizer.joblib")
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            print(f"Warning: Could not load ML models: {e}. Run train.py first.")

router_ml = RouterModel()