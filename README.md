# 📈 Churn Prediction API

A production-ready **FastAPI** micro-service that predicts customer churn using a pre-trained scikit-learn model.  
It ships with Docker, automated CI/CD to Docker Hub + Railway, and a minimal ML experimentation stack (MLflow).

&nbsp;

| Stack | Why |
|-------|-----|
| **Python 3.10 / FastAPI** | Fast, async-friendly REST framework |
| **scikit-learn 1.5.1** | Trained RandomForest & DecisionTree models |
| **Docker (+ docker-compose)** | Reproducible local + prod builds |
| **GitHub Actions** | Test ➜ Build ➜ Push ➜ Deploy (Railway) |
| **MLflow** | Optional experiment tracking & model registry |

---

## 🌐 Live demo

> ```text
> https://<your-service>.railway.app
> ```
> Once the Railway service is **Running**, this URL becomes active.  
> Use `/docs` for Swagger UI and `/health` for a simple health check.

---

## 📂 Repository layout

.
├── app/ # optional package (e.g. routers, utils/)
├── main.py # FastAPI entry point ← uvicorn main:app
├── model.pkl # pre-trained RandomForest model
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── tests/ # pytest tests (optional)
└── .github/
└── workflows/ci-cd.yml

yaml
Copy
Edit

---

## 🚀 Quick start

### 1. Local development (Python)

```bash
# create and activate a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload          # http://127.0.0.1:8000/docs
2. Local development (Docker)
bash
Copy
Edit
docker build -t churn-prediction .
docker run --rm -p 8000:8000 churn-prediction
# visit http://localhost:8000/docs
3. Local development (docker-compose)
bash
Copy
Edit
docker compose up --build
📡 API endpoints
Method	Path	Description
GET	/health	Simple JSON “ok” check
POST	/predict	Return churn probability for one customer
GET	/docs	Interactive Swagger UI
GET	/redoc	Redoc documentation

Predict request example
bash
Copy
Edit
curl -X POST https://<service>.railway.app/predict \
     -H "Content-Type: application/json" \
     -d '{
           "CreditScore": 600,
           "Geography": "France",
           "Gender": "Female",
           "Age": 40,
           "Tenure": 5,
           "Balance": 50000,
           "NumOfProducts": 2,
           "HasCrCard": 1,
           "IsActiveMember": 1,
           "EstimatedSalary": 70000
         }'
Response

json
Copy
Edit
{
  "prediction": 0,
  "probability": 0.17
}
⚙️ Configuration
Variable	Default	Notes
PORT	8000 (overridden by Railway)	Listening port
MODEL_PATH	model.pkl	Path to the pickle file
ENV	production	Toggle extra logging, etc.

🤖 CI / CD pipeline
GitHub Action (.github/workflows/ci-cd.yml)

Install deps & (optional) run tests

Build image, tag :SHA and :latest

Push to Docker Hub (${{ secrets.DOCKERHUB_USERNAME }}/churn-prediction)

Trigger Railway redeploy (railway redeploy)

Railway pulls the latest tag, injects PORT, and starts the container.

🧪 Running tests
bash
Copy
Edit
pytest -q
📚 MLflow (optional)
bash
Copy
Edit
mlflow ui          # http://localhost:5000
# log experiments in training scripts under ./training/
👐 Contributing
Fork → create a feature branch (git checkout -b feature/awesome)

Commit + push + open a PR

Make sure pytest passes and pre-commit linting is clean.

📝 License
This project is licensed under the MIT License – see LICENSE for details.
