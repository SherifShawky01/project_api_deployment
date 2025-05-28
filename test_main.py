from fastapi.testclient import TestClient
from pathlib import Path
import pandas as pd # Import pandas to load CSV data
from custom_transformers import LabelEncoderTransformer
# Adjust the path to locate the `main` module correctly
# This block assumes your test file is NOT in the same directory as main.py
# and main.py is one level up, or in a sibling directory if this is a subpackage.
# If main.py is in the same directory as the test file, simplify this to:
# from main import app
try:
    from main import app
except ImportError:
    # This might be for CI/CD or if your project structure is like:
    # project_root/
    # ├── main.py
    # └── tests/
    #     └── test_api.py
    # If main.py is directly in the project root relative to where you run tests,
    # you might need sys.path adjustments or rely on a proper package structure.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # Add parent directory to path
    from main import app

client = TestClient(app)

def test_home():
    """Test the home endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Hand Gesture Recognition API. Use /docs for Swagger UI."}

def test_health():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy"}

def test_predict():
    """Test the predict endpoint with valid data by loading a sample from CSV."""
    # Load your dataset to get a real sample input
    try:
        df = pd.read_csv("hand_landmarks_data2.csv")
    except FileNotFoundError:
        print("Error: 'hand_landmarks_data2.csv' not found. Please ensure it's in the test directory or accessible path.")
        return # Or raise an error if this test is critical

    # Get a sample row for testing. Pick the first row or any row you want to test.
    # Exclude the 'label' column as it's the target, not an input feature.
    sample_data_series = df.iloc[0].drop("label")

    # Convert the pandas Series to a dictionary, which is the expected format for JSON input
    input_data = sample_data_series.to_dict()

    # Now, send this realistic data to the predict endpoint
    response = client.post("/predict", json=input_data)

    assert response.status_code == 200
    assert "predicted_class_index" in response.json() # Changed based on assumed main.py output
    assert "predicted_class_name" in response.json() # Changed based on assumed main.py output
    assert "confidence_scores" in response.json()
    assert isinstance(response.json()["predicted_class_index"], int)
    assert isinstance(response.json()["predicted_class_name"], str)
    assert isinstance(response.json()["confidence_scores"], list)
    # You might want to add more assertions here, e.g.,
    # assert len(response.json()["confidence_scores"]) == number_of_unique_classes


def test_predict_invalid_data_type():
    """Test the predict endpoint with invalid data type (e.g., string instead of float)."""
    # This test is largely correct for Pydantic's type validation
    input_data = {
        "x1": "invalid",  # Invalid data type for a numerical feature
        "y1": 0.2,
        # ... you need to fill in ALL expected fields for a complete invalid request
        # If your Pydantic model requires many fields, you need to provide them,
        # even if one is invalid, to trigger the correct Pydantic validation.
        # For simplicity in this example, I'll just show the invalid one and assume
        # your Pydantic model for HandLandmarks is comprehensive.
        # Add placeholder values for all other required features if testing schema validation.
    }
    # To make this a truly robust test for an incomplete or malformed request,
    # you might send a partial dictionary or a dictionary with wrong keys.
    # For a type error, one invalid value is enough.
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)
    assert "detail" in response.json()
    assert "value_error" in str(response.json()["detail"]) or "type_error" in str(response.json()["detail"]) # Check for specific error message

def test_predict_missing_data():
    """Test the predict endpoint with missing required data."""
    # Send an incomplete set of features that your Pydantic model expects
    input_data = {
        "x1": 0.1,
        "y1": 0.2,
        # Missing many other features that HandLandmarks expects
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422 # Unprocessable Entity
    assert "detail" in response.json()
    assert "Field required" in str(response.json()["detail"]) # Pydantic's error for missing fields