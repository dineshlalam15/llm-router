from fastapi import FastAPI, HTTPException
from contracts.schemas import RouteRequest, RouteResponse
from .router import predict_provider

app = FastAPI(title="ML based LLM Router API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/route", response_model=RouteResponse)
def route_query(request: RouteRequest):
    """
    Accepts a natural language query and returns the optimal LLM provider 
    based on the trained Logistic Regression model. predict_provider() handles 
    the TF-IDF vectorization, ML prediction, confidence checking,  missing API key fallback logic.
    """
    # try:
    return predict_provider(request.query)
    # except RuntimeError as e:
    #     raise HTTPException(status_code=503, detail=str(e))