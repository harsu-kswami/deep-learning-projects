from models import *  # Import all model classes
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, validator, Field
import numpy as np
import pickle
import os
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("house_price_predictor")

# Initialize FastAPI app with professional configuration
app = FastAPI(
    title="Neural Network House Price Predictor",
    description="Enterprise-grade real estate valuation system using 7 advanced neural network approaches built from scratch",
    version="2.0.0",
    docs_url="/documentation",
    redoc_url="/api-docs",
    openapi_tags=[
        {
            "name": "prediction",
            "description": "Property price prediction operations",
        },
        {
            "name": "system",
            "description": "System health and model management",
        },
        {
            "name": "analytics",
            "description": "Performance analytics and comparisons",
        }
    ]
)

# Add professional CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific domains in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Professional Pydantic models with comprehensive validation
class PropertyFeatures(BaseModel):
    """
    Property features model with comprehensive validation
    All features represent area-level statistics from California Housing Dataset
    """
    
    median_income: float = Field(
        ..., 
        alias="med_inc",
        ge=0.0, 
        le=20.0,
        description="Area median income in $10K units (e.g., 8.32 = $83,200)",
        example=8.32
    )
    
    house_age: float = Field(
        ...,
        ge=0.0,
        le=100.0, 
        description="Median age of houses in the area (years)",
        example=41.0
    )
    
    average_rooms: float = Field(
        ...,
        alias="ave_rooms",
        ge=1.0,
        le=20.0,
        description="Average number of rooms per house in the area",
        example=6.98
    )
    
    average_bedrooms: float = Field(
        ...,
        alias="ave_bedrms", 
        ge=0.0,
        le=5.0,
        description="Average bedrooms per room ratio in the area",
        example=1.02
    )
    
    population: float = Field(
        ...,
        ge=1.0,
        le=50000.0,
        description="Total population in the census block",
        example=322.0
    )
    
    average_occupancy: float = Field(
        ...,
        alias="ave_occup",
        ge=0.5,
        le=20.0,
        description="Average number of people per household",
        example=2.56
    )
    
    latitude: float = Field(
        ...,
        ge=30.0,
        le=45.0,
        description="Geographic latitude (California range)",
        example=37.88
    )
    
    longitude: float = Field(
        ...,
        ge=-130.0,
        le=-110.0,
        description="Geographic longitude (California range)",
        example=-122.23
    )
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "med_inc": 8.32,
                "house_age": 41.0,
                "ave_rooms": 6.98,
                "ave_bedrms": 1.02,
                "population": 322.0,
                "ave_occup": 2.56,
                "latitude": 37.88,
                "longitude": -122.23
            }
        }

class PredictionResult(BaseModel):
    """Professional prediction result model"""
    approach: str
    predicted_price: str
    predicted_price_numeric: float
    confidence_score: str
    performance_metrics: Dict
    prediction_time_ms: float
    timestamp: str

class AnalysisResponse(BaseModel):
    """Professional analysis response model"""
    status: str
    result: Optional[PredictionResult] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None

class SystemHealth(BaseModel):
    """System health response model"""
    status: str
    uptime: str
    models_loaded: int
    available_approaches: List[str]
    system_info: Dict

# Global deployment data
deployment_data = None
startup_time = datetime.now()

def load_deployment_package() -> None:
    """
    Load all trained models with comprehensive error handling
    """
    global deployment_data
    try:
        model_path = 'all_approaches_model.pkl'
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info("Loading neural network models...")
        
        with open(model_path, 'rb') as f:
            deployment_data = pickle.load(f)
        
        logger.info(f"Successfully loaded {len(deployment_data['models'])} neural network approaches")
        
        # Log model performance summary
        for approach, model_data in deployment_data['models'].items():
            r2_score = model_data['performance']['r2']
            logger.info(f"  - {approach}: R² = {r2_score:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

def get_approach_description(approach: str) -> str:
    """Get professional description for each neural network approach"""
    descriptions = {
        'object_oriented': 'Object-oriented architecture with modular components and class-based design',
        'functional': 'Functional programming paradigm with pure mathematical functions',
        'vectorized': 'High-performance vectorized implementation using optimized NumPy operations',
        'sequential': 'Sequential layer-by-layer processing for intuitive network understanding',
        'gradient_first': 'Gradient-focused implementation emphasizing mathematical correctness',
        'modular': 'Production-ready modular system with separated architectural concerns',
        'incremental': 'Incremental complexity building for educational and research purposes'
    }
    return descriptions.get(approach, "Advanced neural network implementation")

# Prediction function registry
PREDICTION_FUNCTIONS = {
    'object_oriented': lambda X, model_data: model_data['model'].predict(X),
    'functional': lambda X, model_data: forward_propagation_function(X, model_data['model'])[0],
    'vectorized': lambda X, model_data: model_data['model'].predict_vectorized(X),
    'sequential': lambda X, model_data: model_data['model'].predict_sequential(X),
    'gradient_first': lambda X, model_data: model_data['model'].predict_gradient_first(X),
    'modular': lambda X, model_data: model_data['model'].predict(X),
    'incremental': lambda X, model_data: model_data['model'].forward(X)
}

def make_professional_prediction(approach: str, features: np.ndarray) -> PredictionResult:
    """
    Execute prediction with comprehensive error handling and professional response
    """
    if deployment_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural network models not loaded. Please contact system administrator."
        )
    
    if approach not in deployment_data['models']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Neural network approach '{approach}' not available. Available approaches: {list(deployment_data['models'].keys())}"
        )
    
    try:
        model_data = deployment_data['models'][approach]
        
        # Feature preprocessing
        features_scaled = deployment_data['scaler'].transform(features)
        
        # Timed prediction execution
        start_time = time.time()
        prediction_func = PREDICTION_FUNCTIONS[approach]
        prediction = prediction_func(features_scaled, model_data)
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Price conversion and formatting
        predicted_value = float(prediction[0][0] if prediction.ndim > 1 else prediction[0])
        price_thousands = predicted_value * 100
        
        return PredictionResult(
            approach=approach,
            predicted_price=f"${price_thousands:.2f}K",
            predicted_price_numeric=round(price_thousands, 2),
            confidence_score=f"{model_data['performance']['r2']:.1%}",
            performance_metrics=model_data['performance'],
            prediction_time_ms=round(execution_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed for approach '{approach}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction execution failed: {str(e)}"
        )

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Initializing Neural Network House Price Predictor...")
    try:
        load_deployment_package()
        logger.info("System initialization completed successfully")
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise

# Professional API Endpoints

@app.get(
    "/",
    summary="System Information",
    description="Get basic system information and available endpoints",
    tags=["system"]
)
def get_system_info():
    """Professional system information endpoint"""
    return {
        "service": "Neural Network House Price Predictor",
        "version": "2.0.0",
        "status": "operational",
        "models_loaded": len(deployment_data['models']) if deployment_data else 0,
        "available_endpoints": {
            "prediction": "/predict/{approach}",
            "batch_analysis": "/predict-all", 
            "performance_comparison": "/compare",
            "system_health": "/health",
            "documentation": "/documentation",
            "web_interface": "/frontend"
        },
        "neural_network_approaches": list(deployment_data['models'].keys()) if deployment_data else []
    }

@app.get(
    "/health",
    summary="System Health Check",
    description="Comprehensive system health and performance metrics",
    response_model=SystemHealth,
    tags=["system"]
)
def health_check():
    """Professional health check endpoint"""
    if deployment_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural network models not loaded"
        )
    
    uptime = datetime.now() - startup_time
    
    return SystemHealth(
        status="healthy",
        uptime=str(uptime).split('.')[0],  # Remove microseconds
        models_loaded=len(deployment_data['models']),
        available_approaches=list(deployment_data['models'].keys()),
        system_info={
            "startup_time": startup_time.isoformat(),
            "python_version": "3.x",
            "framework": "FastAPI",
            "ml_library": "NumPy (Built from Scratch)"
        }
    )

@app.post(
    "/predict/{approach}",
    summary="Single Approach Prediction",
    description="Generate property price prediction using specified neural network approach",
    response_model=AnalysisResponse,
    tags=["prediction"]
)
def predict_single_approach(
    approach: str,
    property_features: PropertyFeatures
) -> AnalysisResponse:
    """Professional single approach prediction endpoint"""
    try:
        logger.info(f"Processing prediction request using {approach} approach")
        
        # Convert Pydantic model to NumPy array
        features = np.array([[
            property_features.median_income,
            property_features.house_age,
            property_features.average_rooms,
            property_features.average_bedrooms,
            property_features.population,
            property_features.average_occupancy,
            property_features.latitude,
            property_features.longitude
        ]])
        
        result = make_professional_prediction(approach, features)
        
        logger.info(f"Prediction completed successfully: {result.predicted_price}")
        
        return AnalysisResponse(
            status="success",
            result=result,
            request_id=f"pred_{int(time.time())}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction request failed: {str(e)}")
        return AnalysisResponse(
            status="error",
            error_message=str(e),
            request_id=f"err_{int(time.time())}"
        )

@app.post(
    "/predict-all",
    summary="Multi-Approach Analysis",
    description="Generate predictions using all available neural network approaches for comprehensive analysis",
    tags=["prediction", "analytics"]
)
def predict_all_approaches(property_features: PropertyFeatures):
    """Professional multi-approach prediction endpoint"""
    try:
        if deployment_data is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neural network models not loaded"
            )
        
        logger.info("Processing multi-approach prediction analysis")
        
        # Convert to NumPy array
        features = np.array([[
            property_features.median_income,
            property_features.house_age,
            property_features.average_rooms,
            property_features.average_bedrooms,
            property_features.population,
            property_features.average_occupancy,
            property_features.latitude,
            property_features.longitude
        ]])
        
        all_predictions = {}
        successful_predictions = []
        
        # Execute predictions across all approaches
        for approach in deployment_data['models'].keys():
            try:
                result = make_professional_prediction(approach, features)
                all_predictions[approach] = result.dict()
                successful_predictions.append(result.predicted_price_numeric)
                
            except Exception as e:
                logger.warning(f"Prediction failed for {approach}: {str(e)}")
                all_predictions[approach] = {"error": str(e)}
        
        if not successful_predictions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="All prediction approaches failed"
            )
        
        # Calculate summary statistics
        avg_prediction = np.mean(successful_predictions)
        std_prediction = np.std(successful_predictions)
        
        # Find best performing approach
        best_approach = max(
            [k for k in all_predictions.keys() if 'error' not in all_predictions[k]],
            key=lambda x: deployment_data['models'][x]['performance']['r2']
        )
        
        logger.info(f"Multi-approach analysis completed. Best approach: {best_approach}")
        
        return {
            "status": "success",
            "predictions": all_predictions,
            "analysis_summary": {
                "best_approach": best_approach,
                "average_prediction": f"${avg_prediction:.2f}K",
                "price_standard_deviation": f"${std_prediction:.2f}K",
                "successful_approaches": len(successful_predictions),
                "total_approaches": len(deployment_data['models'])
            },
            "input_data": property_features.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-approach analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get(
    "/compare",
    summary="Performance Comparison",
    description="Comprehensive performance analysis and comparison of all neural network approaches",
    tags=["analytics"]
)
def compare_approaches():
    """Professional performance comparison endpoint"""
    if deployment_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural network models not loaded"
        )
    
    logger.info("Generating performance comparison analysis")
    
    try:
        comparison_results = []
        
        for approach, model_data in deployment_data['models'].items():
            performance = model_data['performance']
            
            comparison_results.append({
                "approach": approach.replace('_', ' ').title(),
                "description": get_approach_description(approach),
                "r2_score": performance['r2'],
                "mse": performance['mse'],
                "rmse": np.sqrt(performance['mse']),
                "training_time_seconds": performance.get('time', 0.0),
                "accuracy_percentage": f"{performance['r2']:.1%}",
                "performance_rank": 0  # Will be assigned below
            })
        
        # Sort by R² score (descending)
        comparison_results.sort(key=lambda x: x['r2_score'], reverse=True)
        
        # Assign performance ranks
        for i, result in enumerate(comparison_results):
            result['performance_rank'] = i + 1
        
        best_approach = comparison_results[0]
        
        logger.info(f"Performance comparison completed. Best: {best_approach['approach']}")
        
        return {
            "status": "success",
            "performance_analysis": {
                "best_approach": best_approach['approach'],
                "best_r2_score": best_approach['r2_score'],
                "total_approaches_analyzed": len(comparison_results)
            },
            "detailed_comparison": comparison_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance comparison failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison analysis failed: {str(e)}"
        )

@app.get(
    "/approaches",
    summary="Available Approaches",
    description="List all available neural network approaches with detailed descriptions",
    tags=["system"]
)
def get_available_approaches():
    """Professional approaches listing endpoint"""
    if deployment_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural network models not loaded"
        )
    
    approaches_info = {}
    
    for approach, model_data in deployment_data['models'].items():
        approaches_info[approach] = {
            "name": approach.replace('_', ' ').title(),
            "description": get_approach_description(approach),
            "performance_metrics": model_data['performance'],
            "status": "operational"
        }
    
    return {
        "available_approaches": approaches_info,
        "total_count": len(approaches_info)
    }

@app.get(
    "/frontend",
    response_class=HTMLResponse,
    summary="Web Interface",
    description="Professional web interface for property price prediction",
    tags=["system"]
)
def serve_professional_frontend():
    """Serve the professional HTML frontend"""
    try:
        with open('frontend.html', 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("Frontend HTML file not found")
        return HTMLResponse(
            content="""
            <html>
                <head><title>Frontend Not Found</title></head>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1>Professional Web Interface</h1>
                    <p>Frontend file not found. Please ensure 'frontend.html' exists.</p>
                    <p><a href="/documentation">View API Documentation</a></p>
                </body>
            </html>
            """,
            status_code=404
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Professional error response handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.status_code,
            "error_message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": 500,
            "error_message": "Internal server error. Please contact support.",
            "timestamp": datetime.now().isoformat()
        }
    )

# Professional startup
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Neural Network House Price Predictor API...")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info",
        access_log=True
    )
