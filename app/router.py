import os
import json
from dotenv import load_dotenv
from .model_loader import router_ml
from .schemas import RouteResponse
from router_engine.models.catalog import ModelCatalog

load_dotenv()

def get_confidence_threshold() -> float:
    """Reads confidence threshold dynamically from .env."""
    try:
        return float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    except ValueError:
        return 0.65

def get_default_model() -> str:
    """Reads default model from .env (DEFAULT_MODEL or fallback)."""
    return os.getenv("DEFAULT_MODEL") or os.getenv("FALLBACK_MODEL") or "gpt-4o-mini"

def get_default_provider() -> str:
    """Reads default provider from .env (DEFAULT_PROVIDER or fallback)."""
    return os.getenv("DEFAULT_PROVIDER") or os.getenv("FALLBACK_PROVIDER") or "openai"

_catalog = None
_model_metadata = None

def _get_model_details(model_id: str):
    """Retrieves rich metadata and provider name for a predicted model ID."""
    global _catalog, _model_metadata
    if _catalog is None:
        try:
            _catalog = ModelCatalog()
        except Exception:
            _catalog = None

    if _model_metadata is None:
        meta_path = os.path.join("models_out", "model_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    _model_metadata = json.load(f)
            except Exception:
                _model_metadata = {}
        else:
            _model_metadata = {}

    if _model_metadata and model_id in _model_metadata:
        info = _model_metadata[model_id]
        return info.get("provider", get_default_provider()), info

    if _catalog:
        profile = _catalog.get(model_id)
        if profile:
            return profile.provider, profile.to_dict()

    return get_default_provider(), None

def predict_provider(query: str) -> RouteResponse:
    """
    Evaluates the user query with SentenceTransformer embeddings and routes it
    to the optimal LLM model using an SVM primary classifier with KNN fallback.
    Operates strictly locally with zero LLM API calls.
    """
    if router_ml.embedder is None or router_ml.svm_model is None or router_ml.knn_model is None:
        raise RuntimeError("ML routing models are not loaded in memory.")

    query_clean = query.strip()
    threshold = get_confidence_threshold()
    
    # 1. Generate dense semantic embedding (384-d MiniLM)
    query_emb = router_ml.embedder.encode([query_clean])[0]

    # 2. Primary SVM Inference on Model Targets
    svm_model, svm_conf, svm_probs = router_ml.svm_model.predict([query_emb])

    if svm_conf >= threshold:
        prov, model_info = _get_model_details(svm_model)
        return RouteResponse(
            query=query,
            model=svm_model,
            provider=prov,
            model_info=model_info,
            confidence=round(svm_conf, 4),
            probabilities=svm_probs,
            fallback_triggered=False,
            method="svm_primary"
        )

    # 3. Fallback to KNN neighborhood density
    knn_model, knn_conf, knn_probs = router_ml.knn_model.predict(query_emb)

    if knn_conf >= threshold:
        prov, model_info = _get_model_details(knn_model)
        return RouteResponse(
            query=query,
            model=knn_model,
            provider=prov,
            model_info=model_info,
            confidence=round(knn_conf, 4),
            probabilities=knn_probs,
            fallback_triggered=True,
            method="knn_fallback"
        )

    # 4. Low Confidence Default Model Fallback (from .env)
    def_model = get_default_model()
    def_prov = get_default_provider()
    prov, model_info = _get_model_details(def_model)
    return RouteResponse(
        query=query,
        model=def_model,
        provider=prov or def_prov,
        model_info=model_info,
        confidence=round(svm_conf, 4),
        probabilities=svm_probs,
        fallback_triggered=True,
        method="default_model_fallback"
    )