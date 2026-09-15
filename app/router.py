import os
from dotenv import load_dotenv
from .model_loader import router_ml
from .schemas import RouteResponse

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "litellm")

PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "litellm": "LITELLM_API_KEY"
}

def is_provider_configured(provider_name: str) -> bool:
    """Checks if the predicted provider has a valid API key configured."""
    env_var_name = PROVIDER_API_KEYS.get(provider_name)
    if not env_var_name:
        return True
    api_key = os.getenv(env_var_name)
    return bool(api_key and api_key.strip())

def predict_provider(query: str) -> RouteResponse:
    if router_ml.embedder is None or router_ml.svm_model is None or router_ml.knn_model is None:
        raise RuntimeError("ML routing models are not loaded in memory.")

    query_clean = query.strip()
    
    # 1. Generate dense semantic embedding
    query_emb = router_ml.embedder.encode([query_clean])[0]

    # 2. Primary SVM Inference
    svm_provider, svm_conf, svm_probs = router_ml.svm_model.predict([query_emb])

    # Check 1: Primary SVM high confidence + API key configured
    if svm_conf >= CONFIDENCE_THRESHOLD and is_provider_configured(svm_provider):
        return RouteResponse(
            query=query,
            provider=svm_provider,
            confidence=round(svm_conf, 4),
            probabilities=svm_probs,
            fallback_triggered=False
        )

    # Check 2: Fallback to KNN historical density
    knn_provider, knn_conf, knn_probs = router_ml.knn_model.predict(query_emb)

    if knn_conf >= CONFIDENCE_THRESHOLD and is_provider_configured(knn_provider):
        return RouteResponse(
            query=query,
            provider=knn_provider,
            confidence=round(knn_conf, 4),
            probabilities=knn_probs,
            fallback_triggered=True
        )

    # Check 3: Default Gateway Fallback
    return RouteResponse(
        query=query,
        provider=FALLBACK_PROVIDER,
        confidence=round(svm_conf, 4),
        probabilities=svm_probs,
        fallback_triggered=True
    )