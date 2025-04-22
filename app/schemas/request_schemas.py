# app/schemas/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to analyze for mental health concerns")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I've been feeling really down lately and I'm not sure if life is worth living anymore."
            }
        }

class BatchAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text items to analyze")
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "I've been feeling really down lately.",
                    "I don't want to be here anymore, I'm thinking of ending it all."
                ]
            }
        }

class SelfHarmAssessment(BaseModel):
    risk_level: str = Field(..., description="Risk level determined from self-harm indicators (low, elevated, medium, high)")
    active_intent_indicators: List[str] = Field(default_factory=list, description="List of phrases indicating active intent")
    method_indicators: List[str] = Field(default_factory=list, description="List of phrases indicating methods of self-harm")
    planning_indicators: List[str] = Field(default_factory=list, description="List of phrases indicating planning")
    timeframe_indicators: List[str] = Field(default_factory=list, description="List of phrases indicating timeframe")
    requires_followup: bool = Field(..., description="Whether follow-up is recommended based on self-harm indicators")

class AnalysisResponse(BaseModel):
    category: str = Field(..., description="Classified category (Attempt, Behavior, Ideation, Indicator, Supportive)")
    concerning_content: bool = Field(..., description="Whether content is concerning (non-supportive)")
    confidence: float = Field(..., description="Confidence score of the prediction")
    risk_level: str = Field(..., description="Overall risk level (minimal, low, moderate, high, severe)")
    self_harm_assessment: SelfHarmAssessment = Field(..., description="Detailed self-harm risk assessment")
    processed_text: Optional[str] = Field(None, description="Preprocessed text used for model input")
    recommendation: str = Field(..., description="Recommended action based on risk assessment")

class BatchAnalysisResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Analysis results for each text")