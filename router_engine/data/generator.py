import os
import json
import yaml
from typing import List
from .registry import DatasetRegistry, NormalizedSample

def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

def simulate_llm_evaluation(sample: NormalizedSample) -> str:
    query_lower = sample.query.lower()
    
    # 1. Multimodal / Visual / Analytics / Charts -> Gemini
    if any(k in query_lower for k in ["image", "visual", "dashboard", "chart", "graph", "trend", "report", "metric"]):
        return "gemini"
    # 2. Fast / Short / Customer Sentiment -> OpenAI
    elif sample.domain in ["customer_sentiment", "customer_support"] or len(query_lower) < 80:
        return "openai"
    # 3. Long Context / Deep Strategy / Reasoning -> Claude
    elif len(query_lower) > 200 or any(k in query_lower for k in ["strategy", "analyze", "campaign", "repositioning"]):
        return "claude"
    # 4. Fallback -> LiteLLM
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