import pytest
from router_engine.models.catalog import ModelCatalog
from router_engine.eval.profile_matcher import ProfileMatcher

@pytest.fixture
def matcher():
    catalog = ModelCatalog("configs/models_catalog.yaml")
    return ProfileMatcher(catalog=catalog)

def test_visual_multimodal_matching(matcher):
    query = "Analyze this revenue trend chart and export a dashboard visual summary"
    result = matcher.evaluate(query, domain="data_analytics")
    assert result["provider"] == "gemini"
    assert "gemini" in result["optimal_model"]
    assert "multimodal" in result["selection_reason"].lower() or "visual" in result["selection_reason"].lower()

def test_customer_support_matching(matcher):
    query = "Where is my order? I want a refund on my shipping discount"
    result = matcher.evaluate(query, domain="customer_support")
    assert result["provider"] == "openai"
    assert result["optimal_model"] == "gpt-4o-mini"
    assert "cost-efficiency" in result["selection_reason"].lower() or "support" in result["selection_reason"].lower()

def test_math_stem_matching(matcher):
    query = "Natalia sold clips to 48 of her friends in April, and then sold half as many in May. How many total?"
    result = matcher.evaluate(query, domain="metrics_reasoning")
    assert result["provider"] == "openai"
    assert result["optimal_model"] == "gpt-4o"
    assert "math" in result["selection_reason"].lower() or "stem" in result["selection_reason"].lower()

def test_deep_strategy_matching(matcher):
    query = """
    Develop a comprehensive 5-year B2B SaaS market repositioning strategy for an enterprise
    fintech platform facing commoditization from open-source alternatives. Analyze organizational
    trade-offs, architectural decoupling risks, pricing migration matrices, and create a board proposal.
    """ * 2
    result = matcher.evaluate(query, domain="business_strategy")
    assert result["provider"] == "claude"
    assert result["optimal_model"] == "claude-3-5-sonnet"
    assert "reasoning" in result["selection_reason"].lower() or "strategy" in result["selection_reason"].lower()
