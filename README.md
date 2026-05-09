# High-Protein Food Recommender API 🥩🥗

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)

A production-ready Machine Learning REST API and Single-Page Web Application. This project takes a user's target macro-nutritional goals (Calories, Protein, Fat, Carbs, Iron, Vitamin C) and uses a **k-Nearest Neighbors (kNN)** algorithm to mathematically recommend the closest matching high-protein real-world foods.

### 🌐 **Live Demo:** [Macro Matcher UI](https://protein-recommender.onrender.com)
*(Note: Hosted on Render's free tier. The server may take 30-50 seconds to spin up on the very first load).*

---

## 🚀 Features

* **Machine Learning Engine:** Unsupervised k-Nearest Neighbors (kNN) model computing Euclidean distances across a 7-dimensional nutritional space.
* **FastAPI Backend:** High-performance, asynchronous REST API serving the ML model artifacts.
* **Interactive UI:** Clean, responsive HTML/JS/CSS frontend served directly from the backend.
* **Fully Containerized:** Built with Docker for exact environment replication and seamless cloud deployment.
* **MLOps Ready:** Clean separation of data preprocessing, feature building, model training, and API deployment.

---

## 📂 Project Structure

```text
protein_recommender/
│
├── data/
│   ├── raw/                  # Original Kaggle Dataset (Food_Nutrition_Dataset.csv)
│   └── processed/            # Cleaned data (cleaned_food_data.csv)
│
├── models/
│   ├── scaler.pkl            # Trained StandardScaler artifact
│   └── knn_model.pkl         # Trained NearestNeighbors model artifact
│
├── src/                      # Source code for ML Pipeline
│   ├── data/
│   │   └── preprocess.py     # Data cleaning and handling missing values
│   ├── features/
│   │   └── build_features.py # Feature engineering and scaling
│   └── models/
│       └── train.py          # kNN model training logic
│
├── scripts/                  
│   ├── run_pipeline.py       # Orchestrator to run the entire ML pipeline end-to-end
│   └── test_fastapi.py       # FastAPI application and endpoint logic
│
├── index.html                # Frontend User Interface (Macro Matcher)
├── Dockerfile                # Blueprint for containerization
├── requirements.txt          # Production Python dependencies
└── README.md                 # Project documentation
```

---

## 🛠️ Local Setup & Installation

To run this project locally, you can use the pre-configured Docker image.

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/protein-recommender-api.git](https://github.com/YOUR_GITHUB_USERNAME/protein-recommender-api.git)
cd protein-recommender-api
```

**2. Build the Docker Image**
```bash
docker build -t protein-api .
```

**3. Run the Container**
```bash
docker run -p 8000:8000 protein-api
```

**4. Access the Application**
* **Web UI:** Open your browser and go to `http://localhost:8000`
* **Interactive API Docs (Swagger):** Go to `http://localhost:8000/docs`

---

## 🔌 API Endpoint Usage

The backend exposes a POST endpoint that accepts standard JSON requests.

**Endpoint:** `POST /recommend`

**Request Body:**
```json
{
  "calories": 250,
  "protein": 30,
  "carbs": 10,
  "fat": 5,
  "iron": 2,
  "vitamin_c": 0
}
```

**Response Payload:**
```json
{
  "target_request": { ... },
  "recommendations": [
    {
      "rank": 1,
      "food_name": "Chicken Breast, grilled",
      "category": "Meats",
      "mathematical_distance": 0.4321,
      "macros": {
        "calories": 165.0,
        "protein": 31.0,
        "carbs": 0.0,
        "fat": 3.6
      }
    },
    ...
  ]
}
```

---

## 🧠 Model Training Details

If you wish to retrain the model on new data:
1. Ensure your `requirements-dev.txt` is installed.
2. Place the new raw CSV in `data/raw/`.
3. Run the pipeline orchestrator:
   ```bash
   python scripts/run_pipeline.py
   ```
This will automatically clean the data, recalculate the scaling, train the new model, and overwrite the `.pkl` artifacts in the `models/` directory.

---
