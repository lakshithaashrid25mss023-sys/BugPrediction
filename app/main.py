from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Software Bug Prediction API",
    description="Predicts whether a software module may contain a defect."
)

# Load the trained Random Forest model
model_file = Path("models/bug_model.pkl")
saved_data = joblib.load(model_file)

model = saved_data["model"]
features = saved_data["features"]


class ModuleMetrics(BaseModel):
    LOC_TOTAL: float = Field(..., ge=0, description="Total lines of code")
    CYCLOMATIC_COMPLEXITY: float = Field(..., ge=0)
    BRANCH_COUNT: float = Field(..., ge=0)
    NUM_OPERATORS: float = Field(..., ge=0)
    NUM_OPERANDS: float = Field(..., ge=0)


@app.get("/")
def home():
    return {"message": "Software Bug Prediction API is running."}


@app.get("/health")
def health():
    return {"status": "ok", "model": "Random Forest"}


@app.post("/predict")
def predict_bug(metrics: ModuleMetrics):
    # Convert the Swagger input into a table with the same feature order as training
    input_data = pd.DataFrame(
        [metrics.model_dump()],
        columns=features
    )

    prediction = int(model.predict(input_data)[0])
    defect_probability = float(model.predict_proba(input_data)[0][1])

    if prediction == 1:
        result = "Defect likely"
    else:
        result = "No defect likely"

    return {
        "prediction": result,
        "defect_probability": round(defect_probability, 4),
        "note": "This is a prediction, not a guarantee."
    }