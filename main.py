# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from pathlib import Path
from custom_transformers import LabelEncoderTransformer
import pandas as pd

# NEW: Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app and logger
app = FastAPI(
    title="Hand Gesture Recognition API",
    description="API for real-time hand gesture classification.",
    version="1.0.0",
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CORS Configuration ---
# IMPORTANT: For development/debugging, we'll allow all origins ("*").
# For production, you MUST replace "*" with the specific URL(s) of your frontend(s).
# For example, if your GitHub Pages URL is https://your-username.github.io/your-repo-name/
# origins = [
#     "https://your-username.github.io",
#     "https://your-username.github.io/your-repo-name"
# ]
origins = [
    "*"  # Allows all origins. Use ONLY for development/debugging.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies to be included in cross-origin requests
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, PUT, DELETE, OPTIONS)
    allow_headers=["*"],            # Allow all headers
)
# --- End CORS Configuration ---


# Define base directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "SVM_pipeline.pkl" # Or "Random Forest_pipeline.pkl"

# Load the pre-trained model pipeline and label encoder
def load_model_and_encoder():
    try:
        loaded_data = joblib.load(MODEL_PATH)
        model_pipeline = loaded_data["pipeline"]
        label_encoder = loaded_data["label_encoder"]
        logger.info("Model and LabelEncoder loaded successfully.")
        return model_pipeline, label_encoder
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        raise
    except KeyError as e:
        logger.error(f"Error loading model components from {MODEL_PATH}: Missing key {e}. "
                     "Ensure the pipeline was saved as {'pipeline': ..., 'label_encoder': ...}.")
        raise

model_pipeline, label_encoder = load_model_and_encoder()

# Define input data structure (Make sure this matches your training data columns!)
class GesturePredictRequest(BaseModel):
    # Assuming 21 landmarks, x and y coordinates (42 features)
    x1: float; y1: float; x2: float; y2: float; x3: float; y3: float; x4: float; y4: float; x5: float; y5: float;
    x6: float; y6: float; x7: float; y7: float; x8: float; y8: float; x9: float; y9: float; x10: float; y10: float;
    x11: float; y11: float; x12: float; y12: float; x13: float; y13: float; x14: float; y14: float; x15: float; y15: float;
    x16: float; y16: float; x17: float; y17: float; x18: float; y18: float; x19: float; y19: float; x20: float; y20: float;
    x21: float; y21: float;


@app.get("/")
async def home():
    return {"message": "Welcome to the Hand Gesture Recognition API. Use /docs for Swagger UI."}

@app.get("/health")
async def health_check():
    return {"status": "Healthy"}

@app.post("/predict")
async def predict_gesture(data: GesturePredictRequest):
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])

        prediction_encoded = model_pipeline.predict(input_df)[0]
        predicted_class_name = label_encoder.inverse_transform([prediction_encoded])[0]

        confidence_scores_list = []
        classifier_step = model_pipeline.named_steps.get('classifier')
        if hasattr(classifier_step, 'predict_proba'):
            probabilities = classifier_step.predict_proba(input_df)[0]
            for i, score in enumerate(probabilities):
                class_name = label_encoder.inverse_transform([i])[0]
                confidence_scores_list.append({"class": class_name, "score": round(score, 4)})
        else:
            logger.warning("Classifier does not support predict_proba. Confidence scores not available.")

        result = {
            "predicted_class_index": int(prediction_encoded),
            "predicted_class_name": predicted_class_name,
            "confidence_scores": confidence_scores_list
        }

        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
