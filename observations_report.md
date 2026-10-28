# CLA 4(1) - Observations Report

**Title:** ML Model Deployment Strategies and Model Serving using FastAPI  
**Dataset:** Titanic - Machine Learning from Disaster (`train.csv`)

## Model training
The labeled dataset was split into training and testing sets using an 80:20 split. Missing numeric values were imputed using the median. Missing categorical values were imputed using the most frequent value, and categorical columns were one-hot encoded. A Random Forest classifier was trained and saved with its preprocessing steps as `model.pkl` using Joblib.

**Test accuracy:** Fill in the value printed by `train_model.py`.

## Observations

1. **Online inference vs Batch inference:** Online inference predicts one record at a time, usually when a request arrives. Batch inference predicts many records together and is useful for scheduled or bulk processing.
2. **Streaming inference vs Batch inference:** Streaming inference processes each incoming record continuously as it arrives. Batch inference collects a group of records and processes them together.
3. **Why FastAPI is used:** FastAPI makes it straightforward to expose a trained model through REST endpoints, validates request data with Pydantic, and provides interactive Swagger documentation at `/docs`.
4. **Why reuse the same trained model:** Reusing the saved model keeps predictions consistent across online, batch, streaming, and API inference and avoids unnecessary retraining.

## Evidence to include
Add screenshots of online inference, batch inference, streaming inference, successful `/health` and `/predict` tests, and one invalid request rejected by validation.
