from fastapi import FastAPI, HTTPException
from .schemas import RouteRequest, RouteResponse
from .router import predict_provider

app = FastAPI(
    title="ML-Based LLM Router API",
    description="Dynamic routing service utilizing dense SentenceTransformer embeddings and an SVM/KNN architecture."
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/route", response_model=RouteResponse)
def route_query(request: RouteRequest):
    """
    Accepts a natural language query and routes it to the optimal LLM provider
    using an SVM primary classifier with a KNN fallback mechanism.
    """
    try:
        return predict_provider(request.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))