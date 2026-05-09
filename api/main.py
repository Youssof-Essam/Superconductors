import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Superconductivity Prediction API")

MLFLOW_TRACKING_URI = "http://mlflow_server:5000"
MODEL_NAME = "superconductivity_svr_model"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None

@app.on_event("startup")
def load_model():
    global model,FEATURE_NAMES
    for i in range(5):
    
        try:
            model_uri = f"models:/{MODEL_NAME}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            FEATURE_NAMES = model.feature_names_in_.tolist()
            logger.info("Successfully recovered feature names from model.")
            logger.info(f"Successfully loaded model from {model_uri}")
            break
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            time.sleep(5)
            continue
        
            

class Features(BaseModel):
    data: List[float]

@app.post("/predict")
async def predict(features: Features):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([features.data], columns=FEATURE_NAMES)
        prediction = model.predict(df)
        return {"critical_temp": float(prediction[0])}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    

@app.get("/health")
def health_check():
    return {"status": "ready", "model_loaded": model is not None}