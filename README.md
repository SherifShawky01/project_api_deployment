

---

# ✋ Hand Gesture Recognition System and Maze Running

This project implements a real-time **hand gesture recognition system** using machine learning, served via a RESTful API, and integrated into a web-based frontend to control external applications (e.g., a ball in a game) using specific hand gestures captured through a webcam.

---

## 🎯 Project Scope

The primary goal is to develop an **end-to-end solution** for hand gesture recognition:

* **Data Collection & Model Training:**
  Gather hand landmark data and train a robust ML model.

* **Model Serving:**
  Deploy the trained model as a RESTful API.

* **Frontend Integration:**
  Build a web interface that captures landmarks and interacts with the backend.

* **Monitoring & Deployment:**
  Monitor the system and deploy it to a cloud platform for scalability.

---

## 📁 Repository Structure

This project uses a **two-repository approach** for separation of concerns:

### 🔬 Research Repository

* Houses experiment scripts, notebooks, and preprocessing code.
* Includes **MLflow** tracking for experiments.
* Contains `model_comparison.csv` for performance insights.

### 🚀 Production Repository (This Repository)

* Focuses on production code, deployment, and monitoring.
* Contains:

  * `FastAPI` application
  * `Dockerfile`
  * `Unit tests`
  * Monitoring config (Prometheus + Grafana)

---

## 🧠 Model Training & Selection

* Conducted in the research repo using **MLflow**.
* Multiple models evaluated on **accuracy, precision, recall, and F1-score**.
* **Best model**: `SVM_pipeline.pkl`, a Support Vector Machine pipeline with preprocessing.
* Selection recorded in `model_comparison.csv`.

---

## 🌐 Model Serving (FastAPI)

A **RESTful API** is built using FastAPI:

* **Endpoint:** `/predict`

* **Input:** JSON with 21 hand landmarks (x, y for each).

* **Validation:** Uses **Pydantic** models.

* **Model:** Loads `SVM_pipeline.pkl` and `LabelEncoder` using `joblib`.

* **Response:**

  * Predicted gesture
  * Confidence scores for all classes

* **CORS:** Enabled for cross-origin access from the frontend.

---

## 🧪 Unit Testing

Tests are located in `tests/test_main.py` (or `test_main.py` in root):

* Tests for `/`, `/health`, and `/predict`.
* Covers:

  * Valid/invalid input handling
  * Model loading
  * Output structure and response values

---

## 🐳 Containerization (Docker)

The application is **containerized** using Docker:

### `Dockerfile` Includes:

* Base Python image
* App code & dependencies
* `requirements.txt` installation
* Port exposure
* Command to run app via **uvicorn**

---

## 📊 Monitoring Metrics

Key metrics are monitored via **Prometheus** and **Grafana**:

### 🔍 1. Model Metric: Prediction Confidence

* **Why:** Measures live model reliability.
* **How:** Log confidence scores from `/predict`.

### 📉 2. Data Metric: Input Distribution Drift

* **Why:** Detects changes in incoming hand landmark patterns.
* **How:** Track stats (mean, variance) of specific coordinates.

### 🚀 3. Server Metric: Request Latency

* **Why:** Indicates API responsiveness.
* **How:** Expose request durations via FastAPI's metrics.

---

## 📈 System Monitoring Setup

Orchestrated using **Docker Compose**:

### `docker-compose.yaml`:

* Defines:

  * FastAPI app
  * Prometheus
  * Grafana

### Monitoring Files:

* `monitoring/prometheus.yml`: Prometheus targets
* `monitoring/datasource.yml`: Grafana-Prometheus link
* `monitoring/dashboard.yml` or `.json`: Dashboard setup

📸 **Grafana Dashboard Screenshot:**
*(Insert screenshot showing Prediction Confidence, Input Drift, and Latency)*

---

## 🚀 Deployment (CI/CD)

Uses **GitHub Actions** to automate build and deploy:

### `.github/workflows/docker-image.yml`:

* Builds Docker image
* Pushes to container registry (e.g., Docker Hub)
* Deploys to cloud (e.g., Railway.app)

Frontend is deployed separately (e.g., **GitHub Pages**).

---

## 💻 Frontend Integration

### Key Files:

#### `api-call.js`

* Captures landmarks via **MediaPipe**.
* Sends POST request to `/predict` with flattened landmarks.
* Receives predicted gesture and maps it to arrow keys.

#### `mp.js`

* Handles webcam feed and MediaPipe setup.
* Calls `getPredictedLabel()` from `api-call.js`.

#### `keyboard.js`

* Simulates key events (`triggerArrowKey`) to control in-game actions.

---

## 🎬 Demo Video

https://github.com/user-attachments/assets/668f40ae-f375-4405-b5bc-702a2b4eae0e





---

