import json
import yaml
from .registry import DatasetRegistry, NormalizedSample

def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def simulate_llm_evaluation(sample: NormalizedSample) -> str:
    """
    Simulates the ULab api_calling_evaluation.py step. 
    In production, this would fire parallel API calls to OpenAI, Claude, etc., 
    calculate Quality/Cost/Latency, and return the winner.
    """
    query_lower = sample.query.lower()
    
    # Heuristic simulation for label generation
    if sample.domain == "customer_sentiment" or "short" in query_lower:
        return "openai" # Fast, cheap categorization
    elif len(query_lower) > 200 or "strategy" in query_lower or "analyze" in query_lower:
        return "claude" # Complex reasoning / long context
    elif "image" in query_lower or "visual" in query_lower or "dashboard" in query_lower:
        return "gemini" # Multimodal
    else:
        return "litellm" # Abstract fallback

def generate_routing_data(config_path: str = "configs/datasets.yaml"):
    config = load_yaml(config_path)
    
    # 1. Automated Ingestion & Normalization
    raw_samples = DatasetRegistry.ingest_all(config['datasets'])
    
    # 2. Assign Ground-Truth Labels via Evaluation
    evaluated_samples = []
    for sample in raw_samples:
        winner = simulate_llm_evaluation(sample)
        sample.provider_label = winner
        evaluated_samples.append(sample)
        
    # 3. Save as JSONL (ULab standard format)
    out_path = config['output_path']
    with open(out_path, 'w') as f:
        for s in evaluated_samples:
            f.write(json.dumps({"query": s.query, "provider": s.provider_label}) + "\n")
    print(f"Generated {len(evaluated_samples)} routing samples at {out_path}")