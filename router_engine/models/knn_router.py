import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from sklearn.neighbors import KNeighborsClassifier

class KNNRouter:
    def __init__(self, n_neighbors: int = 5, weights: str = "distance", metric: str = "cosine"):
        """
        K-Nearest Neighbors Router.
        
        :param n_neighbors: Number of neighboring query embeddings to inspect.
        :param weights: 'distance' weights closer neighbors more heavily than 'uniform'.
        :param metric: 'cosine' is standard for semantic sentence embeddings.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )

    def train(self, X: Union[np.ndarray, List[List[float]]], y: List[str]):
        """Fits the KNN index on dense embeddings and provider labels."""
        X_mat = np.array(X)
        self.model.fit(X_mat, y)

    def predict(self, query_embedding: Union[np.ndarray, List[float]]) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts the optimal provider, confidence, and class probabilities for a query vector.
        
        :return: (predicted_provider, confidence_score, probability_dict)
        """
        if self.model is None:
            raise RuntimeError("KNN Router model has not been trained or loaded.")

        vec = np.array(query_embedding).reshape(1, -1)
        
        # Calculate vote probabilities across the K nearest neighbors
        probs = self.model.predict_proba(vec)[0]
        classes = self.model.classes_

        best_idx = int(np.argmax(probs))
        predicted_model = str(classes[best_idx])
        confidence = float(probs[best_idx])
        
        prob_dict = {str(classes[i]): round(float(probs[i]), 4) for i in range(len(classes))}

        return predicted_model, confidence, prob_dict

    def find_nearest_examples(self, query_embedding: Union[np.ndarray, List[float]], k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves nearest neighbor distances and metadata."""
        """
        Helper method to retrieve the K most similar historical queries
        and their distances for explainability and inspection.
        """
        vec = np.array(query_embedding).reshape(1, -1)
        distances, indices = self.model.kneighbors(vec, n_neighbors=k)
        
        neighbors_info = []
        for dist, idx in zip(distances[0], indices[0]):
            neighbors_info.append({
                "index": int(idx),
                "distance": round(float(dist), 4),
                "similarity": round(float(1.0 - dist), 4) if self.metric == "cosine" else None
            })
        return neighbors_info

    def save(self, path: str):
        """Serializes and saves the fitted KNN index to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "metric": self.metric
        }, path)

    def load(self, path: str):
        """Loads a persisted KNN index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"KNN model file not found at: {path}")
        
        data = joblib.load(path)
        if isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            self.n_neighbors = data.get("n_neighbors", 5)
            self.weights = data.get("weights", "distance")
            self.metric = data.get("metric", "cosine")
        else:
            self.model = data