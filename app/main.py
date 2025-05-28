from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Initialize the app and logger
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained Random Forest model
# main.py  or  app/app.py
from pathlib import Path
import joblib, fastapi, os

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"        # absolute path
model = joblib.load(MODEL_PATH)

app = fastapi.FastAPI()
# … routes …

model = joblib.load(MODEL_PATH)
logger.info("Random Forest model loaded successfully.")
from joblib import load

def load_model():
    """
    Load the trained model from the specified path.

    Returns:
        model: The loaded model
    """
    try:
        model = load(MODEL_PATH)
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        raise

# Input data structure for predictions
class PredictRequest(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Churn Prediction API. Use /docs for Swagger UI."}

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "Healthy"}

# Predict endpoint
@app.post("/predict")
def predict(input_data: PredictRequest):
    try:
        logger.info("Received prediction request.")
        
        # Convert input data to model-compatible format
        input_array = np.array([
            input_data.CreditScore,
            input_data.Geography == "France",
            input_data.Geography == "Spain",
            input_data.Gender == "Male",
            input_data.Age,
            input_data.Tenure,
            input_data.Balance,
            input_data.NumOfProducts,
            input_data.HasCrCard,
            input_data.IsActiveMember,
            input_data.EstimatedSalary,
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0, 1]

        result = {
            "prediction": int(prediction),
            "probability_of_churn": round(probability, 4),
        }

        logger.info(f"Prediction result: {result}")
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}
