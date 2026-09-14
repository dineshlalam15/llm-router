import os
import json
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from router_engine.models.embeddings import Embedder
from router_engine.models.svm_router import SVMRouter
from router_engine.models.knn_router import KNNRouter

def train_pipeline(config_path: str = "configs/router.yaml", dataset_config_path: str = "configs/datasets.yaml"):
    with open(config_path, "r") as f:
        router_cfg = yaml.safe_load(f)
    with open(dataset_config_path, "r") as f:
        ds_cfg = yaml.safe_load(f)

    data_file = ds_cfg.get("output_path", ds_cfg.get("pipeline_settings", {}).get("output_routing_dataset", "data/processed/routing_dataset.jsonl"))

    print(f"📥 Loading labeled data from: {data_file}")
    queries, labels = [], []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["query"])
            labels.append(item["provider"])

    # 1. Train / Test Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        queries,
        labels,
        test_size=router_cfg.get("training", {}).get("test_size", 0.2),
        random_state=router_cfg.get("training", {}).get("random_state", 42),
        stratify=labels
    )

    # 2. Dense Embeddings
    print("🧠 Encoding queries with SentenceTransformer...")
    embedder = Embedder(router_cfg.get("embedding", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    X_train = embedder.encode(X_train_text)
    X_test = embedder.encode(X_test_text)

    # 3. Train SVM
    print("⚙️ Fitting SVMRouter...")
    svm = SVMRouter(kernel=router_cfg.get("training", {}).get("svm_kernel", "rbf"))
    svm.train(X_train, y_train)

    # 4. Train KNN
    print("⚙️ Fitting KNNRouter...")
    knn = KNNRouter(
        n_neighbors=router_cfg.get("training", {}).get("knn_neighbors", 5),
        weights=router_cfg.get("training", {}).get("knn_weights", "distance"),
        metric=router_cfg.get("training", {}).get("knn_metric", "cosine")
    )
    knn.train(X_train, y_train)

    # 5. Evaluate
    svm_preds = [svm.predict([vec])[0] for vec in X_test]
    knn_preds = [knn.predict(vec)[0] for vec in X_test]

    print("\n📊 SVM Test Accuracy:", round(accuracy_score(y_test, svm_preds), 4))
    print("📊 KNN Test Accuracy:", round(accuracy_score(y_test, knn_preds), 4))
    print("\nKNN Classification Report:\n", classification_report(y_test, knn_preds))

    # 6. Save both models
    model_dir = router_cfg.get("persistence", {}).get("model_dir", "models_out/")
    svm.save(os.path.join(model_dir, "svm.joblib"))
    knn.save(os.path.join(model_dir, "knn.joblib"))
    print(f"✅ Saved SVM and KNN models into '{model_dir}'")

def main():
    parser = argparse.ArgumentParser(description="LLM Router CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train SVM and KNN models")
    train_parser.add_argument("--config", type=str, default="configs/router.yaml")

    args = parser.parse_args()
    if args.command == "train":
        train_pipeline(args.config)

if __name__ == "__main__":
    main()