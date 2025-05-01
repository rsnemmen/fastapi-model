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
from pydantic import BaseModel
import numpy as np # Assuming your model expects numpy arrays
import os # To build file paths reliably

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
class InputFeatures(BaseModel):
    # --- binary categorical ----
    Phone_Service: bool           = Field(..., alias="Phone Service")
    Online_Security: bool         = Field(..., alias="Online Security")
    Online_Backup: bool           = Field(..., alias="Online Backup")
    Premium_Tech_Support: bool    = Field(..., alias="Premium Tech Support")

    # --- multi-class categorical ---
    Contract: Literal["Month-to-month", "One year", "Two year"]

    # --- numeric ---
    Number_of_Referrals: int      = Field(..., alias="Number of Referrals")
    Tenure_in_Months: int         = Field(..., alias="Tenure in Months")
    Monthly_Charge: float         = Field(..., alias="Monthly Charge")
    Satisfaction_Score: int       = Field(..., alias="Satisfaction Score")

    # Example for providing example data in the docs
    class Config:
        allow_population_by_field_name = True   # so you can send either style

        schema_extra = {
            "example": {
                "Phone Service": true,
                "Online Security": false,
                "Online Backup": true,
                "Premium Tech Support": false,
                "Contract": "One year",
                "Number of Referrals": 3,
                "Tenure in Months": 27,
                "Monthly Charge": 72.6,
                "Satisfaction Score": 4
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
    """
    Predicts customer churn based on input features.

    Takes customer characteristics as input and returns a churn prediction (0 or 1).
    """
    # 1. Convert Pydantic model to the format your model expects
    #    Scikit-learn models usually expect a 2D array-like structure (e.g., list of lists or NumPy array)
    #    The order of features MUST match the order used during training!
    feature_values = [
        features.Phone_Service,
        features.Online_Security,
        features.Online_Backup,
        features.Premium_Tech_Support,
        features.Contract,
        features.Number_of_Referrals,
        features.Tenure_in_Months,
        features.Monthly_Charge,
        features.Satisfaction_Score
    ]
    # Convert to 2D NumPy array (as scikit-learn models expect samples in rows)
    input_data = np.array([feature_values])

    # 2. Make prediction
    try:
        prediction_result = model.predict(input_data)
        # If you want probabilities: prediction_proba = model.predict_proba(input_data)
    except Exception as e:
        # Handle potential errors during prediction
        # You might want to raise an HTTPException for client errors
        # or log server errors.
        print(f"Error during prediction: {e}")
        # Example of returning an error response (optional)
        # from fastapi import HTTPException
        # raise HTTPException(status_code=500, detail="Prediction failed.")
        # For now, let's return a default or error indicator if needed,
        # but ideally, handle this more robustly.
        # For simplicity here, we might just let it fail if predict errors.
        # A better approach is specific error handling.

    # 3. Format the output
    # Assuming model.predict returns an array like [0] or [1]
    predicted_class = int(prediction_result[0])

    # If using predict_proba, you might extract the probability for the positive class
    # positive_class_proba = float(prediction_proba[0][1]) # Assuming 1 is the positive class

    # Return the result conforming to the PredictionOutput Pydantic model
    return PredictionOutput(churn_prediction=predicted_class)

# --- (Optional) Add other endpoints, e.g., for model info ---
@app.get("/model_info")
async def get_model_info():
    """Returns information about the loaded model."""
    # You might want to return model parameters, type, etc.
    # Be careful not to expose sensitive information.
    return {"model_type": str(type(model)), "model_path": MODEL_PATH}