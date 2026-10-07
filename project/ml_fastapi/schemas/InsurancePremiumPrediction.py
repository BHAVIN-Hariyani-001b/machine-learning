from pydantic import BaseModel, Field
from typing import Dict


class PredictionResult(BaseModel):
    predicted_category: str = Field(
        ...,
        description="The predicted insurance premium category",
        example="High"
    )
    confidence: float = Field(
        ...,
        description="Model's confidence score for the predicted class (range: 0 to 1)",
        example=0.8432
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all possible classes",
        example={"Low": 0.01, "Medium": 0.15, "High": 0.84}
    )


class PredictionResponse(BaseModel):
    message: str = Field(..., description="Result status message")
    status_code: int = Field(..., description="HTTP-style status code")
    prediction: PredictionResult