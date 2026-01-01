from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (OK for assignment)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Logistic Regression API is running"}

@app.post("/predict")
def predict(features: dict):
    data = np.array(list(features.values())).reshape(1, -1)
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return {"prediction": int(prediction[0])}
