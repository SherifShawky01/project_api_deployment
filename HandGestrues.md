# Hand Gesture Recognition - Final Project

## Overview
This project focuses on recognizing hand gestures using machine learning models trained on a custom dataset of hand landmarks. Multiple classifiers were evaluated, and their performances were compared to identify the best-performing model for the task.

## Dataset
The dataset consists of hand landmark data extracted from images, where:
- **Features**: `x` and `y` coordinates of hand landmarks.
- **Target**: Gesture labels.

### Preprocessing Steps:
1. **Removal of `z`-coordinate features** (not required for the task).
2. **Scaling**: Min-Max scaling was applied to `x` and `y` coordinates.
3. **Label Encoding**: Gesture labels were encoded to integers.
4. **Stratified Splitting**: Data was split into training and test sets (90/10).

---

## Models Evaluated
The following machine learning models were trained and tested:
1. **Support Vector Machine (SVM)**
2. **Logistic Regression**
3. **Random Forest**
4. **XGBoost**
5. **Decision Tree**

All models were trained with their respective hyperparameters tuned for optimal performance.

---

## Results
The models were evaluated based on their accuracy, precision, recall, and F1-score. Below is a summary of the results:

| **Model**              | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-------------------------|--------------|---------------|------------|--------------|
| SVM                    | **0.97**     | **0.97**      | **0.97**   | **0.97**     |
| Logistic Regression     | 0.89         | 0.89          | 0.89       | 0.89         |
| Random Forest           | 0.85         | 0.85          | 0.85       | 0.85         |
| XGBoost                 | 0.72         | 0.72          | 0.72       | 0.72         |
| Decision Tree           | 0.65         | 0.66          | 0.63       | 0.63         |

### Performance Comparison Chart
![Model Performance Comparison](results/model_comparison.png)

---

## Best Model: **Support Vector Machine (SVM)**
The SVM model outperformed all others with the highest accuracy, precision, recall, and F1-score. It is the recommended model for deployment.

---

## MLflow Integration
All experiments, metrics, and artifacts were logged using MLflow. You can view the experiment runs and metrics by connecting to the MLflow server.

---

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
