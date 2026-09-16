import os
import json
import yaml
from typing import List, Optional
from .registry import DatasetRegistry, NormalizedSample
from router_engine.models.catalog import ModelCatalog
from router_engine.eval.profile_matcher import ProfileMatcher

def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

class DatasetGenerator:
    """
    Generates the routing training dataset by ingesting benchmark queries
    and evaluating them against the market LLM profiles catalog.
    """
    def __init__(
        self,
        config_path: str = "configs/datasets.yaml",
        catalog_path: str = "configs/models_catalog.yaml"
    ):
        self.config_path = config_path
        self.config = load_yaml(config_path)
        self.catalog_path = catalog_path
        self.catalog = ModelCatalog(catalog_path)
        self.matcher = ProfileMatcher(catalog=self.catalog)

    def run(self) -> str:
        # 1. Ingest raw benchmark queries from Hugging Face
        print("📥 Ingesting benchmark queries from configured datasets...")
        raw_samples = DatasetRegistry.ingest_all(self.config.get('datasets', []))
        print(f"📊 Ingested {len(raw_samples)} raw benchmark samples.")

        # 2. Evaluate against market LLM catalog profiles
        print("🧠 Evaluating queries against market LLM profiles (capabilities, latency, pricing)...")
        evaluated_records = []
        for sample in raw_samples:
            if sample.optimal_model:
                # Direct Empirical Preference (Human Battle Winner)
                profile = self.catalog.get(sample.optimal_model)
                prov = profile.provider if profile else "openai"
                record = {
                    "query": sample.query,
                    "provider": prov,
                    "model": sample.optimal_model,
                    "domain": sample.domain,
                    "selection_reason": f"Empirically determined winner ({sample.optimal_model}) from human preference battle dataset.",
                    "model_profile": profile.to_dict() if profile else {},
                    "candidate_scores": {}
                }
                evaluated_records.append(record)
            else:
                # Profile-Matched Evaluation
                eval_result = self.matcher.evaluate(
                    query=sample.query,
                    domain=sample.domain
                )
                sample.provider_label = eval_result["provider"]
                sample.optimal_model = eval_result["optimal_model"]
                sample.candidate_evals = eval_result["candidate_scores"]

                record = {
                    "query": sample.query,
                    "provider": sample.provider_label,
                    "model": sample.optimal_model,
                    "domain": sample.domain,
                    "selection_reason": eval_result["selection_reason"],
                    "model_profile": eval_result["model_profile"],
                    "candidate_scores": eval_result["candidate_scores"]
                }
                evaluated_records.append(record)

        # 3. Resolve destination path and ensure directories exist
        out_path = self.config.get('pipeline_settings', {}).get(
            'output_routing_dataset', 'data/processed/routing_dataset.jsonl'
        )
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # 4. Save formatted JSONL
        with open(out_path, 'w', encoding="utf-8") as f:
            for rec in evaluated_records:
                f.write(json.dumps(rec) + "\n")

        print(f"✅ Generated {len(evaluated_records)} profile-evaluated routing samples at {out_path}")
        return out_path

def generate_routing_data(
    config_path: str = "configs/datasets.yaml",
    catalog_path: str = "configs/models_catalog.yaml"
) -> str:
    """Functional wrapper for DatasetGenerator."""
    generator = DatasetGenerator(config_path=config_path, catalog_path=catalog_path)
    return generator.run()