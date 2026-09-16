import pytest
from router_engine.models.catalog import ModelCatalog, LLMModelProfile

def test_catalog_loading():
    catalog = ModelCatalog("configs/models_catalog.yaml")
    models = catalog.get_all()
    assert len(models) >= 6

    # Verify Claude
    claude = catalog.get("claude-3-5-sonnet")
    assert claude is not None
    assert claude.provider == "claude"
    assert claude.input_cost_per_million == 3.00
    assert claude.output_cost_per_million == 15.00
    assert claude.max_input_tokens == 200000

    # Verify GPT-4o Mini
    mini = catalog.get("gpt-4o-mini")
    assert mini is not None
    assert mini.provider == "openai"
    assert mini.cost_tier == "economy"
    assert mini.latency_tier == "ultra_fast"

    # Verify Gemini
    gemini = catalog.get("gemini-1-5-pro")
    assert gemini is not None
    assert gemini.multimodal is True
    assert gemini.max_input_tokens == 2000000

def test_catalog_by_provider():
    catalog = ModelCatalog("configs/models_catalog.yaml")
    openai_models = catalog.get_by_provider("openai")
    assert len(openai_models) >= 2
    ids = [m.id for m in openai_models]
    assert "gpt-4o" in ids
    assert "gpt-4o-mini" in ids
