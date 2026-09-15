import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from sklearn.svm import SVC

class SVMRouter:
    def __init__(self, kernel: str = "rbf", probability: bool = True, C: float = 1.0):
        self.kernel = kernel
        self.probability = probability
        self.C = C
        self.model = SVC(kernel=self.kernel, probability=self.probability, C=self.C, random_state=42)

    def train(self, X: Union[np.ndarray, List[List[float]]], y: List[str]):
        X_mat = np.array(X)
        self.model.fit(X_mat, y)

    def predict(self, query_embeddings: Union[np.ndarray, List[List[float]]]) -> Tuple[str, float, Dict[str, float]]:
        if self.model is None:
            raise RuntimeError("SVM Router model has not been trained or loaded.")

        X_mat = np.array(query_embeddings)
        if len(X_mat.shape) == 1:
            X_mat = X_mat.reshape(1, -1)

        probs = self.model.predict_proba(X_mat)[0]
        classes = self.model.classes_

        best_idx = int(np.argmax(probs))
        predicted_provider = str(classes[best_idx])
        confidence = float(probs[best_idx])

        prob_dict = {str(classes[i]): round(float(probs[i]), 4) for i in range(len(classes))}

        return predicted_provider, confidence, prob_dict

    def save(self, path: str):
        """Serializes and saves the fitted SVM model to disk."""
        # Ensure target directory exists before writing
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Loads a persisted SVM model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"SVM model file not found at: {path}")
        self.model = joblib.load(path)