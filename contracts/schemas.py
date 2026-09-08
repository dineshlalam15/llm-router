from pydantic import BaseModel, Field
from typing import Dict, Optional

class RouteRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The user's input query.")

class RouteResponse(BaseModel):
    query: str
    provider: str
    confidence: float
    probabilities: Dict[str, float]
    fallback_triggered: bool