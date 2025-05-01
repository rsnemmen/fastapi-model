"""
This sets up a mini web server that:

*   Starts by loading your machine learning model.
*   Listens for incoming web requests at specific URLs (`/`, `/predict`).
*   For prediction requests (`POST /predict`):
    *   Validates the incoming customer data.
    *   Uses the loaded model to make a churn prediction.
    *   Sends the prediction back as a structured JSON response.
*   Provides automatic interactive documentation (`/docs`).
"""

import joblib
from fastapi import FastAPI
import numpy as np # Assuming your model expects numpy arrays
import os # To build file paths reliably
import pandas as pd

# pydantic types
from typing import Literal
from pydantic import BaseModel, Field, conint
from enum import Enum

# --- Configuration ---
# Determine the absolute path to the model file
# This makes the script work regardless of where you run it from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

# --- Load the Model ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    # Exit or handle appropriately if the model is essential
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Create FastAPI App ---
app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")

# --- Define Input Data Structure using Pydantic ---
# Pydantic models define the expected data shape, types, and perform validation.
# Replace these feature names with the actual features your model expects.
# 
# CAREFUL: You should use exactly the same value types and spellings present in 
# the training data!
class YesNo(str, Enum):
    yes = "Yes"
    no  = "No"

class InputFeatures(BaseModel):
    # --- binary categorical ----
    Phone_Service: YesNo           = Field(..., alias="Phone Service")
    Online_Security: YesNo         = Field(..., alias="Online Security")
    Online_Backup: YesNo           = Field(..., alias="Online Backup")
    Premium_Tech_Support: YesNo    = Field(..., alias="Premium Tech Support")

    # --- multi-class categorical ---
    Contract: Literal["Month-to-Month", "One Year", "Two Year"]

    # --- numeric ---
    Number_of_Referrals: int      = Field(..., alias="Number of Referrals")
    Tenure_in_Months: int         = Field(..., alias="Tenure in Months")
    Monthly_Charge: float         = Field(..., alias="Monthly Charge")

    # Example for providing example data in the docs
    class Config:
        populate_by_name = True   # so you can send either style

        json_schema_extra = {
            "example": {
                "Phone Service": "Yes",
                "Online Security": "No",
                "Online Backup": "Yes",
                "Premium Tech Support": "No",
                "Contract": "One Year",
                "Number of Referrals": 3,
                "Tenure in Months": 27,
                "Monthly Charge": 72.6,
            }
        }

# --- Define Output Data Structure using Pydantic ---
class PredictionOutput(BaseModel):
    # Example: Assuming binary classification (0 or 1)
    churn_prediction: int # Or float if predict_proba, or str if class names
    # You could also add probabilities if your model provides them
    # churn_probability: Optional[float] = None

# --- API Root Endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API"}


# --- Prediction Endpoint ---
# Define the HTTP method (POST) and the path (/predict)
# response_model ensures the output conforms to PredictionOutput and helps docs
@app.post("/predict", response_model=PredictionOutput)
async def predict_churn(features: InputFeatures):
    # keep the column names that the model was trained with
    df = pd.DataFrame([features.model_dump(by_alias=True)])

    try:
        # predict
        pred = int(model.predict(df)[0])
    except Exception as e:
        print(f"Error during prediction: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

    # return
    return PredictionOutput(churn_prediction=pred)


# --- (Optional) Add other endpoints, e.g., for model info ---
@app.get("/model_info")
async def get_model_info():
    """Returns information about the loaded model."""
    # You might want to return model parameters, type, etc.
    # Be careful not to expose sensitive information.
    return {"model_type": str(type(model)), "model_path": MODEL_PATH}