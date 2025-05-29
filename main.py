from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from pathlib import Path
from custom_transformers import LabelEncoderTransformer # Ensure this is correctly imported

# Initialize the FastAPI app and logger
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS
# For development, you can allow all origins, methods, and headers.
# In production, you should restrict 'allow_origins' to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define base directories
BASE_DIR = Path(__file__).resolve().parent
# Make sure this path is correct relative to where you run main.py
# and that 'SVM_pipeline.pkl' is the name of your saved file.
MODEL_PATH = BASE_DIR / "models" / "SVM_pipeline.pkl"

# Load the pre-trained SVM model
# This function will now return a tuple: (pipeline, label_encoder)
def load_model_and_encoder(): # Renamed function for clarity
    """
    Load the trained model pipeline and label encoder from the specified path.

    Returns:
        tuple: (model_pipeline, label_encoder)
    """
    try:
        loaded_data = joblib.load(MODEL_PATH)
        # Extract the pipeline and label_encoder from the loaded dictionary
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

# Load the model pipeline and label encoder globally when the app starts
model_pipeline, label_encoder = load_model_and_encoder() # Assign to two separate variables

# Define input data structure (Make sure these match your training data columns!)
# This model has 20 features (x_0 to x_19). Your training data has 42 (x1,y1...x21,y21).
# YOU MUST ADJUST THIS TO MATCH YOUR MODEL'S EXPECTED INPUT FEATURES.
# If your model expects x1, y1, x2, y2, etc., then your Pydantic model must reflect that.
class GesturePredictRequest(BaseModel):
    # Your Pydantic model for 21 landmarks (x, y coordinates)
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float
    x5: float
    y5: float
    x6: float
    y6: float
    x7: float
    y7: float
    x8: float
    y8: float
    x9: float
    y9: float
    x10: float
    y10: float
    x11: float
    y11: float
    x12: float
    y12: float
    x13: float
    y13: float
    x14: float
    y14: float
    x15: float
    y15: float
    x16: float
    y16: float
    x17: float
    y17: float
    x18: float
    y18: float
    x19: float
    y19: float
    x20: float
    y20: float
    x21: float
    y21: float
    # If your model also uses 'z' coordinates, add them here:
    # z1: float
    # ...
    # z21: float


# Define routes
@app.get("/")
def home():
    """
    Home endpoint to verify the API is running.
    """
    return {"message": "Welcome to the Hand Gesture Recognition API. Use /docs for Swagger UI."}

@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "Healthy"}

@app.post("/predict", response_model=dict) # Added response_model for clarity
def predict(input_data: GesturePredictRequest):
    """
    Predict endpoint for hand gesture classification.

    Args:
        input_data (GesturePredictRequest): Input feature values.

    Returns:
        dict: Prediction result and confidence score.
    """
    try:
        logger.info("Received prediction request.")

        # Convert input Pydantic model to a Pandas DataFrame
        # This is crucial for your ColumnTransformer based preprocessor
        input_df = pd.DataFrame([input_data.model_dump()])

        confidence_scores_list = []
        predicted_class_name = "unknown" # Default value
        predicted_encoded_label = None # Initialize for scope

        classifier_step = model_pipeline.named_steps.get('classifier')

        # Check if the classifier supports predict_proba and has classes_ attribute
        if hasattr(classifier_step, 'predict_proba') and hasattr(classifier_step, 'classes_'):
            probabilities = classifier_step.predict_proba(input_df)[0]

            # Find the index of the maximum probability
            max_proba_index = np.argmax(probabilities)
            # Get the encoded label corresponding to the max probability
            predicted_encoded_label = classifier_step.classes_[max_proba_index]
            # Inverse transform to get the human-readable class name
            predicted_class_name = label_encoder.inverse_transform([predicted_encoded_label])[0]

            # Populate confidence scores list
            for i, score in enumerate(probabilities):
                encoded_label_for_proba = classifier_step.classes_[i]
                class_name = label_encoder.inverse_transform([encoded_label_for_proba])[0]
                confidence_scores_list.append({"class": class_name, "score": round(score, 4)})
        else:
            logger.warning("Classifier does not support predict_proba or does not have classes_. Confidence scores not available.")
            # Fallback to direct predict if predict_proba is not available
            prediction_encoded = model_pipeline.predict(input_df)[0]
            predicted_class_name = label_encoder.inverse_transform([prediction_encoded])[0]
            predicted_encoded_label = prediction_encoded # Assign for consistency

        result = {
            "action": predicted_class_name, # This will now be the class with the highest probability
            "predicted_class_index": int(predicted_encoded_label) if predicted_encoded_label is not None else None,
            "confidence_scores": confidence_scores_list,
        }

        logger.info(f"Prediction result: {result}")
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        # Return HTTP 500 for internal server errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
