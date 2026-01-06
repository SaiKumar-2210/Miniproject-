from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.api.inference import InferenceEngine

app = FastAPI(title="Agri-Price Prediction API", version="1.0")

# Input: Allow React Frontend (Vite defaults to 5173)
origins = [
    "http://localhost:5173",
    "http://localhost:3000", # Potential alternative
    "*" # For development convenience
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine
# We initialize it once at startup
engine = InferenceEngine()

class PredictionRequest(BaseModel):
    commodity: str
    district: str
    # date: Optional[str] = None # Future extension

class PredictionResponse(BaseModel):
    commodity: str
    district: str
    modal_price: float
    volatility: float = 0.0
    price_min: float = 0.0
    price_max: float = 0.0
    risk_level: str = "Unknown"

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    try:
        result = engine.predict(request.commodity, request.district)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return {
            "commodity": request.commodity,
            "district": request.district,
            "modal_price": result.get("modal_price", 0.0),
            "volatility": result.get("volatility", 0.0),
            "price_min": result.get("price_min", 0.0),
            "price_max": result.get("price_max", 0.0),
            "risk_level": result.get("risk_level", "Unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
