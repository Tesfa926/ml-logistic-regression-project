from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/predict")
def predict(features: list[float]):
    data = np.array(features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return {"prediction": int(prediction[0])}
