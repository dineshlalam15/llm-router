import os
import yaml
from router_engine.models.embeddings import Embedder
from router_engine.models.svm_router import SVMRouter
from router_engine.models.knn_router import KNNRouter

class RouterModelLoader:
    def __init__(self, config_path: str = "configs/router.yaml"):
        self.config_path = config_path
        self.embedder = None
        self.svm_model = None
        self.knn_model = None
        self.config = {}
        self._load_all()

    def _load_all(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_cfg_path = os.path.join(base_dir, self.config_path)

        if not os.path.exists(full_cfg_path):
            full_cfg_path = self.config_path

        with open(full_cfg_path, "r") as f:
            self.config = yaml.safe_load(f)

        model_name = self.config.get("embedding", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = Embedder(model_name=model_name)

        model_dir = self.config.get("persistence", {}).get("model_dir", "models_out/")
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(base_dir, model_dir)

        self.svm_model = SVMRouter(kernel=self.config.get("training", {}).get("svm_kernel", "rbf"))
        self.knn_model = KNNRouter(
            n_neighbors=self.config.get("training", {}).get("knn_neighbors", 5),
            weights=self.config.get("training", {}).get("knn_weights", "distance"),
            metric=self.config.get("training", {}).get("knn_metric", "cosine")
        )

        self.svm_model.load(os.path.join(model_dir, "svm.joblib"))
        self.knn_model.load(os.path.join(model_dir, "knn.joblib"))
        print("SUCCESS: Loaded SentenceTransformer, SVM Router, and KNN Router.")

router_ml = RouterModelLoader()