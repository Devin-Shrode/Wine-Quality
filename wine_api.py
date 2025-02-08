
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load trained LightGBM model
model_filename = "lightgbm_wine_quality_model.pkl"
model = joblib.load(model_filename)

# Initialize FastAPI app
app = FastAPI(title="Wine Quality Prediction API", description="Predicts wine quality based on input features.")

# Define request model for validation
class WineFeatures(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Quality Prediction API"}

@app.post("/predict")
def predict_quality(wine: WineFeatures):
    prediction = model.predict([wine.features])
    return {"predicted_quality": int(prediction[0])}
