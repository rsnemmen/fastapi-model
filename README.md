# FastAPI Model Serving API

This project provides a simple web API built with FastAPI to serve a [pre-trained XGBoost model](https://github.com/rsnemmen/telco-churn). It allows users to send input feature data via a POST request and receive predictions from the model.

## Features

*   Serves a pre-trained XGBoost model.
*   Provides a `/predict` endpoint for making predictions.
*   Uses Pydantic for request data validation.
*   Includes automatic interactive API documentation (Swagger UI and ReDoc).

## Project Structure

```txt
.
├── 📄 main.py: FastAPI application script containing API logic 
│ and Pydantic models
└── 📄 model.joblib: Pre-trained model file w/ scikit-learn
```

## Instructions

(1) Clone the repository.

(2) Install dependencies:

    pip install -r requirements.txt

(3) Run the Server

To start the API server locally, run the following command in your terminal from the project's root directory:

    uvicorn main:app --reload

The server will typically start on `http://127.0.0.1:8000`.

## Usage

**(1) Access API Documentation:**

Once the server is running, open your web browser and navigate to:
*   `http://127.0.0.1:8000/docs` for interactive Swagger UI documentation.
*   `http://127.0.0.1:8000/redoc` for alternative ReDoc documentation.

You can view the expected request format and test the `/predict` endpoint directly from the Swagger UI.

**(2) Send a Prediction Request (Example using `curl`):**

You can send a `POST` request to the `/predict` endpoint with the input features in the JSON body. Open a new terminal (while the server is running) and use a command like `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "Phone Service":"Yes",
           "Online Security":"Yes",
           "Online Backup":"Yes",
           "Premium Tech Support":"No",
           "Contract":"Month-to-Month",
           "Number of Referrals":0,
           "Tenure in Months":4,
           "Monthly Charge":120
         }'
```

The server will respond with a JSON object containing the prediction, such as:

```json
{"churn_prediction": 1}
```

## Challenges

While writing this API, I spent most of my time figuring out how to specify the API input features vs those used to train the model. For the particular case study I used here, the scikit-learn pipeline was actually fed with 50 features, 42 of which are noninformative and eventually discarded. 

I had to retrain the model to discard those features earlier in the training, because FastAPI wants exactly the same features you fed the model—even if they were discarded in the pipeline.