from fastapi import FastAPI, Query, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from mlflow import MlflowClient
import mlflow.pyfunc
import pandas as pd
from typing import Union

app = FastAPI()

# Set up Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


def fetch_latest_model_name() -> str:
    client = MlflowClient()
    models = list(client.search_registered_models())
    if not models:
        raise HTTPException(status_code=404, detail="No registered models found.")
    return models[0].name


def load_production_model(model_name: str):
    try:
        return mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.get("/predict/")
def model_output(
    sepal_length: float = Query(...),
    sepal_width: float = Query(...),
    petal_length: float = Query(...),
    petal_width: float = Query(...)
):
    model_name = fetch_latest_model_name()
    model = load_production_model(model_name)

    input_df = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }])

    try:
        prediction = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"prediction": prediction[0]}
