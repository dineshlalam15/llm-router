import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class NormalizedSample:
    query: str
    domain: str
    provider_label: Optional[str] = None

class DatasetAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_and_normalize(self) -> List[NormalizedSample]:
        dataset_name = self.config['name']
        config_name = self.config.get('config_name', None)
        split = self.config.get('split', 'train')
        text_column = self.config.get('text_column', 'text')
        domain = self.config.get('domain', 'general')
        sample_size = self.config.get('sample_size', 100)
        filter_keywords = self.config.get('filter_keywords', [])
        category_filter = self.config.get('category_filter', None)

        print(f"📥 Loading dataset: {dataset_name}" + (f" ({config_name})" if config_name else ""))

        # FIX: Pass config_name if specified for datasets like GSM8K
        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split, cache_dir="data/cache")
        else:
            ds = load_dataset(dataset_name, split=split, cache_dir="data/cache")

        # Category filtering (e.g. AG News)
        if category_filter:
            col = category_filter['column']
            val = category_filter['value']
            ds = ds.filter(lambda x: x[col] == val)

        # Keyword filtering
        if filter_keywords:
            def matches_keywords(example):
                txt = str(example.get(text_column, "")).lower()
                return any(kw.lower() in txt for kw in filter_keywords)
            ds = ds.filter(matches_keywords)

        # Shuffle and downsample
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