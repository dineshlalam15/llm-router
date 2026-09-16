import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datasets import load_dataset

import json

@dataclass
class NormalizedSample:
    """
    Normalized Sample: A Structured data container
        1. query: The actual question. 
        2. domain: The category/topic the query falls into. 
        3. provider_label: Brand Provider of the model (openai, claude etc.)
        4. optimal_model: Final model that is chosen. 
        5. candidate_evals: Mathematical breakdown of the scores for all the models. 
    """
    query: str
    domain: str
    provider_label: Optional[str] = None
    optimal_model: Optional[str] = None
    candidate_evals: Optional[Dict[str, Any]] = None

class DatasetAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _normalize_prompt(self, raw_prompt: Any) -> str:
        """Extracts the first user turn from text, list, or serialized JSON."""
        if isinstance(raw_prompt, list):
            return str(raw_prompt[0]).strip() if raw_prompt else ""
        if isinstance(raw_prompt, str):
            raw_prompt = raw_prompt.strip()
            if raw_prompt.startswith("[") and raw_prompt.endswith("]"):
                try:
                    parsed = json.loads(raw_prompt)
                    if isinstance(parsed, list) and parsed:
                        return str(parsed[0]).strip()
                except Exception:
                    pass
            return raw_prompt
        return str(raw_prompt).strip()

    def _map_model(self, raw_name: str, model_mapping: Dict[str, str]) -> Optional[str]:
        """Resolves raw model name to catalog model ID via exact or prefix matching."""
        if not raw_name:
            return None
        raw_clean = raw_name.strip()
        raw_lower = raw_clean.lower()
        if raw_clean in model_mapping:
            return model_mapping[raw_clean]
        if raw_lower in model_mapping:
            return model_mapping[raw_lower]
        for k, v in model_mapping.items():
            if raw_lower.startswith(k.lower()) or k.lower() in raw_lower:
                return v
        return None

    def load_and_normalize(self) -> List[NormalizedSample]:
        ingestion_mode = self.config.get('ingestion_mode', 'profile_matched')
        domain = self.config.get('domain', 'general')
        sample_size = self.config.get('sample_size', 100)
        dataset_name = self.config['name']
        config_name = self.config.get('config_name', None)
        split = self.config.get('split', 'train')

        print(f"📥 Loading dataset: {dataset_name}" + (f" ({config_name})" if config_name else "") + f" [mode: {ingestion_mode}]")

        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split, cache_dir="data/cache")
        else:
            ds = load_dataset(dataset_name, split=split, cache_dir="data/cache")

        if ingestion_mode == "direct_preference":
            prompt_col = self.config.get('prompt_column', self.config.get('text_column', 'prompt'))
            model_a_col = self.config.get('model_a_column', 'model_a')
            model_b_col = self.config.get('model_b_column', 'model_b')
            winner_a_col = self.config.get('winner_a_column', 'winner_model_a')
            winner_b_col = self.config.get('winner_b_column', 'winner_model_b')
            winner_tie_col = self.config.get('winner_tie_column', 'winner_tie')
            model_mapping = self.config.get('model_mapping', {})

            ds = ds.shuffle(seed=42)
            samples = []
            for row in ds:
                if winner_tie_col and row.get(winner_tie_col) == 1:
                    continue

                winner_raw = None
                if row.get(winner_a_col) == 1:
                    winner_raw = row.get(model_a_col)
                elif row.get(winner_b_col) == 1:
                    winner_raw = row.get(model_b_col)

                if not winner_raw:
                    continue

                mapped_model = self._map_model(str(winner_raw), model_mapping)
                if not mapped_model:
                    continue

                query_text = self._normalize_prompt(row.get(prompt_col, ""))
                if query_text and len(query_text) >= 5:
                    samples.append(NormalizedSample(query=query_text, domain=domain, optimal_model=mapped_model))
                    if len(samples) >= sample_size:
                        break

            return samples

        else:
            text_column = self.config.get('text_column', 'text')
            filter_keywords = self.config.get('filter_keywords', [])
            category_filter = self.config.get('category_filter', None)

            if category_filter:
                col = category_filter['column']
                val = category_filter['value']
                ds = ds.filter(lambda x: x[col] == val)

            if filter_keywords:
                def matches_keywords(example):
                    txt = str(example.get(text_column, "")).lower()
                    return any(kw.lower() in txt for kw in filter_keywords)
                ds = ds.filter(matches_keywords)

            ds = ds.shuffle(seed=42)
            if len(ds) > sample_size:
                ds = ds.select(range(sample_size))

            samples = []
            for row in ds:
                query_text = str(row.get(text_column, "")).strip()
                if query_text:
                    samples.append(NormalizedSample(query=query_text, domain=domain))

            return samples

class DatasetRegistry:
    @staticmethod
    def ingest_all(dataset_configs: List[Dict[str, Any]]) -> List[NormalizedSample]:
        all_samples = []
        for cfg in dataset_configs:
            if not cfg.get('enabled', True):
                continue
            adapter = DatasetAdapter(cfg)
            samples = adapter.load_and_normalize()
            all_samples.extend(samples)
        return all_samples