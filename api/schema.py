from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    """
    Input schema for prediction request
    """
    features: List[float] = Field(
        ...,
        example=[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
        description="List of numerical feature values in correct order"
    )


class PredictionResponse(BaseModel):
    """
    Output schema for prediction response
    """
    prediction: int = Field(
        ...,
        example=1,
        description="Predicted class label (0 = No Disease, 1 = Disease)"
    )
    confidence: float = Field(
        ...,
        example=0.87,
        description="Prediction confidence score"
    )
    latency_seconds: float = Field(
        ...,
        example=0.0123,
        description="Inference latency in seconds"
    )