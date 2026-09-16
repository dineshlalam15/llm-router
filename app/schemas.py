from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language prompt to evaluate.")

class RouteResponse(BaseModel):
    query: str
    model: str
    provider: str
    model_info: Optional[Dict[str, Any]] = None
    confidence: float
    probabilities: Dict[str, float]
    fallback_triggered: bool
    method: Optional[str] = None