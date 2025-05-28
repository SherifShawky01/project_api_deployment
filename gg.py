import pandas as pd
from custom_transformers import LabelEncoderTransformer
import joblib
import numpy as np
import os # Import os for path handling

# Load the original dataset to get a valid sample
df = pd.read_csv("hand_landmarks_data2.csv")

# Separate features and labels (as done in training script)
X = df.drop(columns=["label"])
y = df["label"]

# Choose a random sample from your original features for testing
# You can pick any row you want for testing
sample_index = 0 # Change this index to test different samples if needed
sample_input_df = X.iloc[[sample_index]] # Get the features for a specific row as a DataFrame

# Get the true label for comparison (optional, but good for verification)
true_label_original = y.iloc[sample_index] # Get the string label directly

# Define the path to the saved model
model_name = "Random Forest"  # Make sure this matches a model you've trained and saved
models_dir = "models"
pipeline_load_path = os.path.join(models_dir, f"{model_name}_pipeline.pkl")

try:
    pipeline_data = joblib.load(pipeline_load_path)
except FileNotFoundError:
    print(f"Error: The file '{pipeline_load_path}' was not found.")
    print("Please ensure the model training script has been run and the pipeline saved to the 'models/' directory.")
    exit()

pipeline = pipeline_data["pipeline"]
label_encoder = pipeline_data["label_encoder"]

# Make predictions
predicted_label_encoded = pipeline.predict(sample_input_df)[0]
predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]

print(f"True Gesture (from CSV): {true_label_original}")
print(f"Predicted Gesture: {predicted_label}")