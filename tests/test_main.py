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
    input_data = {
      "x1": 262.9130402, "y1": 229.5256154,
      "x2": 250.8313293, "y2": 225.6459928,
      "x3": 243.6705551, "y3": 213.6319452,
      "x4": 241.8025589, "y4": 202.8055616,
      "x5": 237.8146591, "y5": 194.2824285,
      "x6": 249.8142014, "y6": 196.3489535,
      "x7": 248.0120773, "y7": 182.3233712,
      "x8": 247.9545135, "y8": 173.1732023,
      "x9": 248.3393555, "y9": 165.7534064,
      "x10": 257.554184, "y10": 194.9684608,
      "x11": 255.6020966, "y11": 179.8632231,
      "x12": 254.7074661, "y12": 169.8857015,
      "x13": 254.1591797, "y13": 161.8072067,
      "x14": 265.0940323, "y14": 196.2305138,
      "x15": 263.2237244, "y15": 181.6690636,
      "x16": 261.8060074, "y16": 172.7557279,
      "x17": 260.586937, "y17": 165.1772625,
      "x18": 272.6696091, "y18": 199.779223,
      "x19": 271.5879593, "y19": 188.6274729,
      "x20": 270.1034775, "y20": 181.5163569,
      "x21": 268.3308792, "y21": 175.4415079
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    # Add assertions for prediction content as well, e.g.:
    assert "predicted_class_name" in response.json()
    assert "confidence_scores" in response.json()
    assert isinstance(response.json()["predicted_class_name"], str)
    assert isinstance(response.json()["confidence_scores"], list)


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
