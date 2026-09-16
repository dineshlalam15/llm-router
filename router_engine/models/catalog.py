import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class LLMModelProfile:
    id: str
    name: str
    provider: str
    company: str
    input_cost_per_million: float
    output_cost_per_million: float
    cost_tier: str
    latency_tier: str
    typical_latency_ms: int
    max_input_tokens: int
    max_output_tokens: int
    multimodal: bool
    suitable_tasks: str
    strengths: List[str] = field(default_factory=list)
    best_domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "company": self.company,
            "pricing": {
                "input_per_million": self.input_cost_per_million,
                "output_per_million": self.output_cost_per_million,
                "tier": self.cost_tier
            },
            "latency": {
                "tier": self.latency_tier,
                "typical_ms": self.typical_latency_ms
            },
            "context": {
                "max_input_tokens": self.max_input_tokens,
                "max_output_tokens": self.max_output_tokens
            },
            "multimodal": self.multimodal,
            "suitable_tasks": self.suitable_tasks,
            "strengths": self.strengths,
            "best_domains": self.best_domains
        }

class ModelCatalog:
    def __init__(self, config_path: str = "configs/models_catalog.yaml"):
        self.config_path = config_path
        self.models: Dict[str, LLMModelProfile] = {}
        self.load(config_path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Catalog file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.models.clear()
        for item in data.get("models", []):
            profile = LLMModelProfile(
                id=item["id"],
                name=item["name"],
                provider=item["provider"],
                company=item.get("company", ""),
                input_cost_per_million=float(item.get("pricing", {}).get("input_per_million", 1.0)),
                output_cost_per_million=float(item.get("pricing", {}).get("output_per_million", 2.0)),
                cost_tier=item.get("pricing", {}).get("tier", "balanced"),
                latency_tier=item.get("latency", {}).get("tier", "fast"),
                typical_latency_ms=int(item.get("latency", {}).get("typical_ms", 500)),
                max_input_tokens=int(item.get("context", {}).get("max_input_tokens", 128000)),
                max_output_tokens=int(item.get("context", {}).get("max_output_tokens", 4096)),
                multimodal=bool(item.get("multimodal", False)),
                suitable_tasks=item.get("suitable_tasks", ""),
                strengths=item.get("strengths", []),
                best_domains=item.get("best_domains", [])
            )
            self.models[profile.id] = profile

    def get_all(self) -> List[LLMModelProfile]:
        return list(self.models.values())

    def get(self, model_id: str) -> Optional[LLMModelProfile]:
        return self.models.get(model_id)

    def get_by_provider(self, provider: str) -> List[LLMModelProfile]:
        return [m for m in self.models.values() if m.provider == provider]

    def get_default_model_for_provider(self, provider: str) -> Optional[LLMModelProfile]:
        models = self.get_by_provider(provider)
        return models[0] if models else None
