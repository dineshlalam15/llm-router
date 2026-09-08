import os
from .model_loader import router_ml
from contracts.schemas import RouteResponse
from train import clean_text

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "litellm")

PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "litellm": "LITELLM_API_KEY"
}

def validate_provider(provider_name: str) -> bool:
    """
    Checks if the predicted provider has a valid API key in the environment.
    Returns True if valid, False if missing, empty, or unknown.
    """
    env_var_name = PROVIDER_API_KEYS.get(provider_name)
    
    if not env_var_name:
        print(f"Error: Provider '{provider_name}' is not mapped in PROVIDER_API_KEYS.")
        return False
        
    api_key = os.getenv(env_var_name)
    return bool(api_key and api_key.strip())


def predict_provider(query: str) -> RouteResponse:
    if router_ml.model is None or router_ml.vectorizer is None:
        raise RuntimeError(f"Model is {type(router_ml.model)}, Vectorizer is {type(router_ml.vectorizer)}")
    query_clean = clean_text(query)
    x_vec = router_ml.vectorizer.transform([query_clean])
    probabilities = router_ml.model.predict_proba(x_vec)[0]
    classes = router_ml.model.classes_
    for i in range(len(classes)):
        probabilities_dict = {classes[i]: round(float(probabilities[i]), 4)}
    best_idx = probabilities.argmax()
    predicted_provider = classes[best_idx]
    confidence = float(probabilities[best_idx])
    fallback_triggered = False

    if confidence < CONFIDENCE_THRESHOLD:
        fallback_triggered = True
        predicted_provider = FALLBACK_PROVIDER
    elif not validate_provider(predict_provider):
        fallback_triggered = True
        print(f"Warning: Rerouting from {classes[best_idx]} due to missing API key.")
        predict_provider=FALLBACK_PROVIDER

    return RouteResponse(
        query=query,
        provider=predicted_provider,
        confidence=confidence,
        probabilities=probabilities_dict,
        fallback_triggered=fallback_triggered
    )

    
