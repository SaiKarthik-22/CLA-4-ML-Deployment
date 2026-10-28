from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Titanic Survival Prediction API", version="1.0")
model = joblib.load("model.pkl")
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str = Field(..., pattern="^(male|female)$")
    Age: Optional[float] = Field(default=None, ge=0, le=100)
    SibSp: int = Field(default=0, ge=0)
    Parch: int = Field(default=0, ge=0)
    Fare: Optional[float] = Field(default=None, ge=0)
    Embarked: Optional[str] = Field(default=None, pattern="^(S|C|Q)$")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(passenger: Passenger):
    row = pd.DataFrame([passenger.model_dump()])[FEATURES]
    prediction = int(model.predict(row)[0])
    return {
        "prediction": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive"
    }
