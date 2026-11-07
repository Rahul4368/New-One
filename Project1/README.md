# 🤖 GitHub Repository Popularity Classifier

> **ML-powered web app** that predicts a GitHub repository's popularity tier using a trained Random Forest model — built with FastAPI + Streamlit.

---

## 📌 Project Overview

Yeh project ek **Machine Learning based Web Application** hai jo **GitHub repositories** ki metrics (jaise stars, forks, issues) dekhkar automatically predict karta hai ki woh repository kitni popular hai.

Model teen categories mein classify karta hai:

| Category | Description |
|----------|-------------|
| 🟡 **Established (100–1k)** | Medium popularity repos |
| 🟠 **High (1k–10k)** | Well-known repos |
| 🔴 **Mega (10k+)** | Highly popular / viral repos |

---

## 🏗️ Project Structure

```
Project1/
│
├── backend/                  ← FastAPI REST API
│   ├── main.py               ← API server with prediction endpoints
│   └── requirements.txt      ← Backend dependencies
│
├── frontend/                 ← Streamlit Web App (UI)
│   ├── app.py                ← Dashboard, Prediction & Model Info pages
│   └── requirements.txt      ← Frontend dependencies
│
├── models/                   ← Trained ML Model & Data
│   ├── model.joblib           ← Trained Random Forest model
│   ├── feature_names.json     ← Input feature names
│   ├── target_labels.json     ← Output class labels
│   ├── metrics.json           ← Model accuracy & metadata
│   └── feature_examples.csv  ← Sample input data (332 rows)
│
└── README.md                 ← This file
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | Scikit-learn — Random Forest Classifier |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend UI** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Model Storage** | Joblib |

---

## 🧠 ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | `RandomForestClassifier` |
| Accuracy | **97.59%** |
| Total Features | **8** |
| Total Classes | **3** |

### Input Features (8 total)

| Feature | Description |
|---------|-------------|
| `stars` | GitHub stars count |
| `forks` | Number of forks |
| `open_issues` | Open issues count |
| `size_kb` | Repo size in KB |
| `repo_age_days` | Age of repo in days |
| `days_since_last_push` | Days since last code push |
| `releases_count` | Number of releases |
| `github_health_pct` | GitHub community health % |

---

## 🚀 How to Run

### Step 1 — Start the Backend (FastAPI)

```bash
cd Project1/backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Backend will be live at: **http://localhost:8000**

### Step 2 — Start the Frontend (Streamlit)

```bash
cd Project1/frontend
python -m streamlit run app.py --server.port 8501
```

Frontend will open at: **http://localhost:8501**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status check |
| `GET` | `/health` | Backend health check |
| `GET` | `/features` | List all input features |
| `GET` | `/metrics` | Model accuracy & metadata |
| `GET` | `/sample` | Sample input data |
| `POST` | `/predict` | 🔮 Make a prediction |

### Example Prediction Request

```json
POST /predict
{
  "features": {
    "stars": 298,
    "forks": 14,
    "open_issues": 0,
    "size_kb": 24274,
    "repo_age_days": 1234,
    "days_since_last_push": 10,
    "releases_count": 5,
    "github_health_pct": 72
  }
}
```

### Example Response

```json
{
  "status": "success",
  "prediction": "0",
  "predicted_label": "Established (100-1k)",
  "probabilities": {
    "Established (100-1k)": 97.0,
    "High (1k-10k)": 3.0,
    "Mega (10k+)": 0.0
  }
}
```

---

## 🖥️ UI Pages

| Page | Description |
|------|-------------|
| 📊 **Dashboard** | Model accuracy, feature count, algorithm overview |
| 🔍 **Prediction** | Enter repo stats — get prediction + probability chart |
| 📈 **Model Information** | Feature list, target labels, full metrics |

---

## 📦 Installation

```bash
# Backend dependencies
pip install fastapi uvicorn scikit-learn pandas numpy joblib pydantic

# Frontend dependencies
pip install streamlit requests pandas
```

---

## 🔧 Requirements

- Python 3.8+
- pip packages (see above)

---

## 👨‍💻 Developed By

**Rahul**
Built using Python | FastAPI | Streamlit | Scikit-learn
