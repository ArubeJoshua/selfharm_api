# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, Any

from app.models.suicide_classifier import SuicidalThoughtClassifier
from app.schemas.request_schemas import (
    TextAnalysisRequest, 
    BatchAnalysisRequest, 
    AnalysisResponse,
    BatchAnalysisResponse,
    SelfHarmAssessment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mental-health-text-analysis-api")

# Initialize the FastAPI application
app = FastAPI(
    title="Mental Health Text Analysis API",
    description="API for detecting concerning mental health content using a trained machine learning model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a dependency for the classifier
def get_classifier():
    return app.state.classifier

# Initialize classifier at startup
@app.on_event("startup")
async def startup_event():
    try:
        app.state.classifier = SuicidalThoughtClassifier()
        logger.info("Successfully loaded Advanced Ensemble classifier model")
    except Exception as e:
        logger.error(f"Failed to load classifier model: {str(e)}")
        raise RuntimeError(f"Could not initialize model: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Mental Health Text Analysis API",
        "description": "Use the /analyze endpoint to analyze text for concerning content",
        "model": "Advanced Ensemble Classifier",
        "documentation": "/docs",
        "categories": ["Attempt", "Behavior", "Ideation", "Indicator", "Supportive"]
    }

@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest, 
    classifier: SuicidalThoughtClassifier = Depends(get_classifier)
):
    """
    Analyze a single text for mental health concerns and risk level.
    
    Returns detailed analysis including:
    - Category classification (Attempt, Behavior, Ideation, Indicator, Supportive)
    - Confidence score
    - Overall risk level
    - Detailed self-harm assessment
    - Recommendation for action
    """
    try:
        # Log the analysis request
        logger.info(f"Analyzing text (length: {len(request.text)})")
        
        # Perform analysis
        result = classifier.predict(request.text)
        
        # Create proper SelfHarmAssessment object
        self_harm_assessment = SelfHarmAssessment(
            risk_level=result["self_harm_assessment"]["risk_level"],
            active_intent_indicators=result["self_harm_assessment"]["active_intent_indicators"],
            method_indicators=result["self_harm_assessment"]["method_indicators"],
            planning_indicators=result["self_harm_assessment"]["planning_indicators"],
            timeframe_indicators=result["self_harm_assessment"]["timeframe_indicators"],
            requires_followup=result["self_harm_assessment"]["requires_followup"]
        )
        
        # Create response with proper typing
        response = AnalysisResponse(
            category=result["category"],
            concerning_content=result["concerning_content"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            self_harm_assessment=self_harm_assessment,
            processed_text=result["processed_text"],
            recommendation=result["recommendation"]
        )
        
        # Log analysis completion
        logger.info(f"Analysis complete. Category: {result['category']}, Risk level: {result['risk_level']}")
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/analyze-batch/", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    classifier: SuicidalThoughtClassifier = Depends(get_classifier)
):
    """
    Analyze multiple text items for mental health concerns.
    
    Returns a list of analysis results for each provided text.
    """
    results = []
    try:
        logger.info(f"Processing batch analysis request with {len(request.texts)} texts")
        
        for i, text in enumerate(request.texts):
            logger.debug(f"Analyzing text {i+1}/{len(request.texts)}")
            
            # Process text
            analysis = classifier.predict(text)
            
            # Create self-harm assessment object
            self_harm_assessment = SelfHarmAssessment(
                risk_level=analysis["self_harm_assessment"]["risk_level"],
                active_intent_indicators=analysis["self_harm_assessment"]["active_intent_indicators"],
                method_indicators=analysis["self_harm_assessment"]["method_indicators"],
                planning_indicators=analysis["self_harm_assessment"]["planning_indicators"],
                timeframe_indicators=analysis["self_harm_assessment"]["timeframe_indicators"],
                requires_followup=analysis["self_harm_assessment"]["requires_followup"]
            )
            
            # Create response object
            response = {
                "text_snippet": text[:50] + "..." if len(text) > 50 else text,
                "analysis": {
                    "category": analysis["category"],
                    "concerning_content": analysis["concerning_content"],
                    "confidence": analysis["confidence"],
                    "risk_level": analysis["risk_level"],
                    "self_harm_assessment": self_harm_assessment.dict(),
                    "recommendation": analysis["recommendation"]
                }
            }
            
            results.append(response)
        
        logger.info(f"Batch analysis complete")
        return BatchAnalysisResponse(results=results)
    
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis error: {str(e)}")

@app.get("/health/")
async def health_check(classifier: SuicidalThoughtClassifier = Depends(get_classifier)):
    """
    Health check endpoint to verify the API is running properly.
    """
    try:
        # Try to make a simple prediction to verify model is working
        test_text = "This is a test."
        result = classifier.predict(test_text)
        
        return {
            "status": "healthy", 
            "model_loaded": True,
            "model_type": "Advanced Ensemble Classifier",
            "categories": ["Attempt", "Behavior", "Ideation", "Indicator", "Supportive"],
            "service": "Mental Health Text Analysis API"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "Mental Health Text Analysis API"
        }

@app.get("/categories/")
async def get_categories():
    """
    Returns information about the classification categories used by the model.
    """
    return {
        "categories": {
            "Attempt": "Content indicating an actual suicide attempt or specific imminent plan",
            "Behavior": "Content describing self-harm behaviors or actions",
            "Ideation": "Content expressing suicidal thoughts or desires",
            "Indicator": "Content showing warning signs or risk factors for self-harm",
            "Supportive": "Content offering help or support to someone in crisis"
        },
        "risk_levels": {
            "severe": "Highest risk level requiring immediate intervention",
            "high": "Significant risk requiring prompt follow-up",
            "moderate": "Concerning content warranting attention",
            "low": "Some concerning elements but lower immediate risk",
            "minimal": "No significant indicators of immediate risk"
        }
    }

# Run the application (if this file is executed directly)
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)