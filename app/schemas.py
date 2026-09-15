from pydantic import BaseModel, Field
from typing import Dict

class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language prompt to evaluate.")

class RouteResponse(BaseModel):
    query: str
    provider: str
    confidence: float
    probabilities: Dict[str, float]
    fallback_triggered: bool