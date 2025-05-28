"""
Train hand gesture recognition models with MLflow logging and save full pipeline.
"""
from custom_transformers import LabelEncoderTransformer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import os # Import os for path handling

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom transformer for label encoding
# class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.encoder = LabelEncoder()

#     def fit(self, y):
#         self.encoder.fit(y)
#         return self

#     def transform(self, y):
#         return self.encoder.transform(y)

#     def inverse_transform(self, y):
#         return self.encoder.inverse_transform(y)

# Load and preprocess the dataset
logger.info("Loading dataset...")
df = pd.read_csv("hand_landmarks_data2.csv")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Preprocessing pipeline
# Filter out 'z' coordinates if they exist and you don't want to scale them
# The original code only scales numerical features that DO NOT start with 'z'
numerical_features = [col for col in X.columns if not col.startswith("z")]
numerical_transformer = MinMaxScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features)
    ]
)

# Encode labels
label_encoder = LabelEncoderTransformer()
y_encoded = label_encoder.fit_transform(y)

# --- Added verification for label encoding ---
unique_labels_in_data = sorted(y.unique())
unique_labels_in_encoder = sorted(label_encoder.encoder.classes_.tolist())

if unique_labels_in_data == unique_labels_in_encoder:
    logger.info(f"LabelEncoder successfully fitted on all {len(unique_labels_in_data)} unique labels.")
    logger.info(f"Unique labels: {unique_labels_in_data}")
else:
    logger.warning("Mismatch in unique labels. This should not happen if `fit_transform(y)` is on full 'y'.")
    logger.warning(f"Labels in data: {unique_labels_in_data}")
    logger.warning(f"Labels in encoder: {unique_labels_in_encoder}")
# --- End of added verification ---

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

# Create a directory for saving models if it doesn't exist
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
logger.info(f"Ensured directory '{models_dir}' exists for saving models.")

# Function to train, log, and save models
def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test, preprocessor, label_encoder, **params):
    """
    Train a model, log results to MLflow, and save pipeline.
    """
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)

        # Combine preprocessing and model into a pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        logger.info(f"Training {model_name}...")
        pipeline.fit(X_train, y_train)

        # Define the path to save the pipeline
        pipeline_save_path = os.path.join(models_dir, f"{model_name}_pipeline.pkl")

        # Save the pipeline and label encoder
        joblib.dump(
            {"pipeline": pipeline, "label_encoder": label_encoder}, # Use the fully fitted label_encoder
            pipeline_save_path
        )
        logger.info(f"Pipeline and LabelEncoder saved to '{pipeline_save_path}'.")


        # Predict and evaluate
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Log the model
        mlflow.sklearn.log_model(pipeline, f"{model_name}_pipeline")

        # Log confusion matrix
        conf_mat = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=label_encoder.encoder.classes_) # Use encoder classes for display
        disp.plot(cmap=plt.cm.Blues) # Add a colormap
        plt.title(f"Confusion Matrix for {model_name}")
        cm_image_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_image_path)
        mlflow.log_artifact(cm_image_path)
        plt.close() # Close the plot to prevent it from displaying unnecessarily

        return {"model": model_name, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Hand Gesture Recognition")

    # Define models
    models = {
        "SVM": (SVC(C=10, kernel="poly", degree=4, gamma="scale", class_weight=None, probability=True), {}), # Added probability=True for potential future use
        "Logistic Regression": (LogisticRegression(max_iter=1000), {"max_iter": 1000}),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), {"n_estimators": 100}),
        "XGBoost": (XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric="mlogloss", use_label_encoder=False), {"n_estimators": 100, "learning_rate": 0.1}), # use_label_encoder=False to suppress warning
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {}),
    }

    results = []
    for model_name, (model, params) in models.items():
        result = train_and_log_model(model_name, model, X_train, y_train, X_test, y_test, preprocessor, label_encoder, **params)
        results.append(result)

    # Generate model comparison
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison.csv", index=False)

    # Plotting comparison
    fig, ax = plt.subplots(figsize=(10, 6)) # Create a figure and an axes object
    results_df.plot(x="model", y=["accuracy", "precision", "recall", "f1_score"], kind="bar", ax=ax)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha="right") # Rotate x-axis labels for better readability
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig("model_comparison.png")
    plt.close() # Close the plot

if __name__ == "__main__":
    main()