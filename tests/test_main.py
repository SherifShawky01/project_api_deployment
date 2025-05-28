from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    """Test the home endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Churn Prediction API. Use /docs for Swagger UI."}

def test_health():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy"}

def test_predict():
    """Test the predict endpoint with valid data."""
    input_data = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 70000.0
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability_of_churn" in response.json()

def test_predict_invalid_data():
    """Test the predict endpoint with invalid data."""
    input_data = {
        "CreditScore": "invalid",  # Invalid data type
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 70000.0
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422  # Unprocessable Entity
