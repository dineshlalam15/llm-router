from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class NormalizedSample:
    query: str
    domain: str
    provider_label: Optional[str] = None

class DatasetLoader:
    def __init__(self, config: dict):
        self.config = config

    def load_and_normalize(self) -> List[NormalizedSample]:
        print(f"Downloading/Loading {self.config['name']} from Hugging Face...")
        ds = load_dataset(self.config['name'], split=self.config['split'], cache_dir="data/cache")
        
        # Shuffle and sample
        ds = ds.shuffle(seed=42).select(range(min(self.config['sample_size'], len(ds))))
        
        samples = []
        for row in ds:
            text = row.get(self.config['text_column'])
            if text and len(text.strip()) > 10:
                samples.append(NormalizedSample(query=text, domain=self.config['domain']))
        return samples

class DatasetRegistry:
    @staticmethod
    def ingest_all(datasets_config: list) -> List[NormalizedSample]:
        all_samples = []
        for ds_cfg in datasets_config:
            if ds_cfg.get("enabled", False):
                loader = DatasetLoader(ds_cfg)
                all_samples.extend(loader.load_and_normalize())
        return all_samples