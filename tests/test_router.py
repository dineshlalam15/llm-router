import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_healthcheck(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data

def test_route_prediction(client):
    response = client.post("/route", json={"query": "Draft a detailed strategic repositioning campaign."})
    assert response.status_code == 200
    data = response.json()
    assert "model" in data and data["model"]
    assert "provider" in data and data["provider"]
    assert "confidence" in data
    assert "probabilities" in data
    assert "fallback_triggered" in data

def test_diverse_model_routing(client):
    # 1. Customer Support -> should route to gpt-4o-mini
    resp_support = client.post("/route", json={"query": "Where is my order? I want a refund on my shipping discount."})
    assert resp_support.status_code == 200
    data_support = resp_support.json()
    assert data_support["model"] == "gpt-4o-mini"
    assert data_support["provider"] == "openai"

    # 2. Strategic copy -> should route to claude-3-5-sonnet
    resp_strategy = client.post("/route", json={"query": "Draft an executive 5-year strategic repositioning proposal for board presentation."})
    assert resp_strategy.status_code == 200
    data_strategy = resp_strategy.json()
    assert data_strategy["model"] == "claude-3-5-sonnet"
    assert data_strategy["provider"] == "claude"

    # 3. Math problem -> should route to gpt-4o
    resp_math = client.post("/route", json={"query": "Natalia sold 48 clips in April and half as many in May. How many clips did she sell in total?"})
    assert resp_math.status_code == 200
    data_math = resp_math.json()
    assert data_math["model"] == "gpt-4o"
    assert data_math["provider"] == "openai"

    # 4. Code Generation -> should route to llama-3-1-70b-instruct
    resp_code = client.post("/route", json={"query": "Write a python function to check if every number in a list is prime."})
    assert resp_code.status_code == 200
    data_code = resp_code.json()
    assert data_code["model"] == "llama-3-1-70b-instruct"
    assert data_code["provider"] == "litellm"

def test_env_fallback_configuration(client, monkeypatch):
    """Verifies that when confidence fails, the router strictly falls back to .env DEFAULT_MODEL and DEFAULT_PROVIDER."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1.01")
    monkeypatch.setenv("DEFAULT_MODEL", "llama-3-1-70b-instruct")
    monkeypatch.setenv("DEFAULT_PROVIDER", "litellm")

    resp = client.post("/route", json={"query": "Any random query guaranteed to trigger fallback"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback_triggered"] is True
    assert data["model"] == "llama-3-1-70b-instruct"
    assert data["provider"] == "litellm"
    assert data["method"] == "default_model_fallback"

def test_dynamic_router_engine_fallback(monkeypatch):
    """Verifies that DynamicRouter engine directly uses .env DEFAULT_MODEL and DEFAULT_PROVIDER."""
    from router_engine.routing.engine import DynamicRouter
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1.01")
    monkeypatch.setenv("DEFAULT_MODEL", "custom-fallback-model")
    monkeypatch.setenv("DEFAULT_PROVIDER", "custom-provider")

    engine = DynamicRouter()
    res = engine.route("Low confidence fallback query")
    assert res["fallback_triggered"] is True
    assert res["model"] == "custom-fallback-model"
    assert res["provider"] == "custom-provider"
    assert res["method"] == "default_model_fallback"

def test_direct_preference_adapter():
    """Verifies that DatasetAdapter correctly normalizes prompts and maps external models."""
    from router_engine.data.registry import DatasetAdapter

    cfg = {
        "name": "test_dataset",
        "ingestion_mode": "direct_preference",
        "model_mapping": {
            "gpt-4-1106-preview": "gpt-4o",
            "claude-3-opus": "claude-3-5-sonnet",
            "llama-3-8b": "llama-3-1-8b-instruct"
        }
    }
    adapter = DatasetAdapter(cfg)

    # Test prompt normalization
    assert adapter._normalize_prompt('["What is machine learning?"]') == "What is machine learning?"
    assert adapter._normalize_prompt(["Tell me a joke."]) == "Tell me a joke."
    assert adapter._normalize_prompt("Simple string query") == "Simple string query"

    # Test model resolution
    assert adapter._map_model("gpt-4-1106-preview", cfg["model_mapping"]) == "gpt-4o"
    assert adapter._map_model("claude-3-opus-20240229", cfg["model_mapping"]) == "claude-3-5-sonnet"
    assert adapter._map_model("unknown-old-model-xyz", cfg["model_mapping"]) is None


