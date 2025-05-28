from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import mlflow
import mlflow.sklearn
from prefect import flow

mlflow.set_experiment("Test2")


@flow
def train():
    # Enable automatic logging
    mlflow.sklearn.autolog()

    # Load dataset
    iris = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target = "species"

    X = iris[features]
    y = iris[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    forest = RandomForestClassifier(random_state=42)

    with mlflow.start_run():
        forest.fit(X_train, y_train)
        predictions = forest.predict(X_test)

        # Manually log metrics
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", acc)


if __name__ == "__main__":
    train()
