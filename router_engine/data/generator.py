import os
import json
import yaml
from typing import List
from .registry import DatasetRegistry, NormalizedSample

def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

def simulate_llm_evaluation(sample: NormalizedSample) -> str:
    """
    Simulates the benchmark evaluation step.
    Assigns routing ground-truth labels based on domain characteristics.
    """
    query_lower = sample.query.lower()
    
    if sample.domain == "customer_sentiment" or "short" in query_lower:
        return "openai"
    elif len(query_lower) > 200 or "strategy" in query_lower or "analyze" in query_lower:
        return "claude"
    elif "image" in query_lower or "visual" in query_lower or "dashboard" in query_lower:
        return "gemini"
    else:
        return "litellm"

class DatasetGenerator:
    """Class interface for dataset generation pipeline."""
    def __init__(self, config_path: str = "configs/datasets.yaml"):
        self.config_path = config_path
        self.config = load_yaml(config_path)

    def run(self) -> str:
        # 1. Automated Ingestion & Normalization
        raw_samples = DatasetRegistry.ingest_all(self.config.get('datasets', []))
        
        # 2. Assign Ground-Truth Labels via Evaluation
        evaluated_samples = []
        for sample in raw_samples:
            sample.provider_label = simulate_llm_evaluation(sample)
            evaluated_samples.append(sample)
            
        # 3. Resolve path and ensure parent directories exist
        out_path = self.config.get('output_path', 'data/processed/routing_dataset.jsonl')
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # 4. Save as JSONL
        with open(out_path, 'w', encoding="utf-8") as f:
            for s in evaluated_samples:
                f.write(json.dumps({"query": s.query, "provider": s.provider_label}) + "\n")
                
        print(f"Generated {len(evaluated_samples)} routing samples at {out_path}")
        return out_path

def generate_routing_data(config_path: str = "configs/datasets.yaml") -> str:
    """Functional wrapper for DatasetGenerator."""
    generator = DatasetGenerator(config_path=config_path)
    return generator.run()