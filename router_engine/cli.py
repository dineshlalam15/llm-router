import os
import json
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from router_engine.models.embeddings import Embedder
from router_engine.models.svm_router import SVMRouter
from router_engine.models.knn_router import KNNRouter
from router_engine.models.catalog import ModelCatalog
from router_engine.data.generator import DatasetGenerator

def generate_data_pipeline(
    config_path: str = "configs/datasets.yaml",
    catalog_path: str = "configs/models_catalog.yaml"
):
    print(f"🔄 Initializing automated dataset generation:")
    print(f"   Datasets Config: {config_path}")
    print(f"   Models Catalog:  {catalog_path}")
    generator = DatasetGenerator(config_path=config_path, catalog_path=catalog_path)
    output_file = generator.run()
    print(f"✅ Labeled dataset successfully generated at: {output_file}")

def train_pipeline(
    config_path: str = "configs/router.yaml",
    dataset_config_path: str = "configs/datasets.yaml",
    catalog_path: str = "configs/models_catalog.yaml"
):
    """Trains dense embeddings, SVM, and KNN fallback routers using empirical dataset."""
    with open(config_path, "r") as f:
        router_cfg = yaml.safe_load(f)
    with open(dataset_config_path, "r") as f:
        ds_cfg = yaml.safe_load(f)

    data_file = ds_cfg.get("pipeline_settings", {}).get(
        "output_routing_dataset", "data/processed/routing_dataset.jsonl"
    )

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Dataset not found at {data_file}. Run 'generate-data' first."
        )

    print(f"📥 Loading labeled data from: {data_file}")
    queries, model_labels, provider_labels = [], [], []
    provider_to_models = {}

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["query"])
            model_id = item.get("model") or item.get("provider")
            prov = item.get("provider", "")
            model_labels.append(model_id)
            provider_labels.append(prov)

            if prov not in provider_to_models:
                provider_to_models[prov] = {}
            provider_to_models[prov][model_id] = provider_to_models[prov].get(model_id, 0) + 1

    # Load catalog metadata for rich runtime routing responses
    catalog = ModelCatalog(catalog_path)
    model_metadata = {}
    for m in catalog.get_all():
        model_metadata[m.id] = {
            "id": m.id,
            "name": m.name,
            "provider": m.provider,
            "company": m.company,
            "pricing": {
                "input_per_million": m.input_cost_per_million,
                "output_per_million": m.output_cost_per_million,
                "tier": m.cost_tier
            },
            "latency": {
                "tier": m.latency_tier,
                "typical_ms": m.typical_latency_ms
            },
            "strengths": m.strengths,
            "best_domains": m.best_domains
        }

    provider_defaults = {}
    for prov, model_counts in provider_to_models.items():
        top_model_id = max(model_counts, key=model_counts.get)
        profile = catalog.get(top_model_id) or catalog.get_default_model_for_provider(prov)
        provider_defaults[prov] = {
            "default_model": top_model_id,
            "model_profile": profile.to_dict() if profile else {}
        }

    # 1. Train / Test Split on Model IDs
    from collections import Counter
    counts = Counter(model_labels)
    can_stratify = min(counts.values()) >= 2 if counts else False

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        queries,
        model_labels,
        test_size=router_cfg.get("training", {}).get("test_size", 0.2),
        random_state=router_cfg.get("training", {}).get("random_state", 42),
        stratify=model_labels if can_stratify else None
    )

    # 2. Dense Embeddings
    print("🧠 Encoding queries with SentenceTransformer...")
    embedder = Embedder(
        router_cfg.get("embedding", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    )
    X_train = embedder.encode(X_train_text)
    X_test = embedder.encode(X_test_text)

    # 3. Train SVM on Model Targets
    print("⚙️ Fitting SVMRouter on LLM model targets...")
    svm = SVMRouter(kernel=router_cfg.get("training", {}).get("svm_kernel", "rbf"))
    svm.train(X_train, y_train)

    # 4. Train KNN on Model Targets
    print("⚙️ Fitting KNNRouter on LLM model targets...")
    knn = KNNRouter(
        n_neighbors=router_cfg.get("training", {}).get("knn_neighbors", 5),
        weights=router_cfg.get("training", {}).get("knn_weights", "distance"),
        metric=router_cfg.get("training", {}).get("knn_metric", "cosine")
    )
    knn.train(X_train, y_train)

    # 5. Evaluate
    svm_preds = [svm.predict([vec])[0] for vec in X_test]
    knn_preds = [knn.predict(vec)[0] for vec in X_test]

    print("\n📊 SVM Model Test Accuracy:", round(accuracy_score(y_test, svm_preds), 4))
    print("📊 KNN Model Test Accuracy:", round(accuracy_score(y_test, knn_preds), 4))
    print("\nSVM Classification Report:\n", classification_report(y_test, svm_preds))

    # 6. Save Artifacts
    model_dir = router_cfg.get("persistence", {}).get("model_dir", "models_out/")
    os.makedirs(model_dir, exist_ok=True)
    svm.save(os.path.join(model_dir, "svm.joblib"))
    knn.save(os.path.join(model_dir, "knn.joblib"))

    # Save model metadata and provider mappings for serving layer
    with open(os.path.join(model_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)
    with open(os.path.join(model_dir, "provider_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(provider_defaults, f, indent=2)

    print(f"✅ Saved model-level SVM, KNN, and model metadata to '{model_dir}'")

def route_query_pipeline(query: str):
    from router_engine.routing.engine import DynamicRouter
    router = DynamicRouter()
    res = router.route(query)
    print("\n🎯 Routing Decision:")
    print(f"   Query:               {res['query']}")
    print(f"   Optimal Model:       {res['model']}")
    print(f"   Provider:            {res['provider']}")
    print(f"   Confidence Score:    {res['confidence']:.4f}")
    print(f"   Routing Method:      {res['method']}")
    print(f"   Fallback Triggered:  {res['fallback_triggered']}")
    if "probabilities" in res and res["probabilities"]:
        print("\n📊 Top Model Probabilities:")
        sorted_probs = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)
        for m, p in sorted_probs[:5]:
            print(f"   - {m:25s}: {p:.4f}")
    print()

def main():
    parser = argparse.ArgumentParser(description="LLM Router CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate-data
    gen_parser = subparsers.add_parser("generate-data", help="Download and generate labeled routing data using models catalog")
    gen_parser.add_argument("--config", type=str, default="configs/datasets.yaml", help="Path to datasets config")
    gen_parser.add_argument("--catalog", type=str, default="configs/models_catalog.yaml", help="Path to models catalog config")

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train SVM and KNN models")
    train_parser.add_argument("--config", type=str, default="configs/router.yaml", help="Path to router config")
    train_parser.add_argument("--dataset-config", type=str, default="configs/datasets.yaml", help="Path to datasets config")
    train_parser.add_argument("--catalog", type=str, default="configs/models_catalog.yaml", help="Path to models catalog config")

    # Subcommand: route
    route_parser = subparsers.add_parser("route", help="Evaluate and route a user query directly from CLI")
    route_parser.add_argument("query", type=str, help="Prompt or query to evaluate")

    args = parser.parse_args()

    if args.command == "generate-data":
        generate_data_pipeline(args.config, args.catalog)
    elif args.command == "train":
        train_pipeline(args.config, args.dataset_config, args.catalog)
    elif args.command == "route":
        route_query_pipeline(args.query)

if __name__ == "__main__":
    main()