import time
import pandas as pd
import joblib

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
model = joblib.load("model.pkl")
records = pd.read_csv("unseen_records.csv").head(10)

print("Streaming inference: processing one passenger per second")
for index, row in records.iterrows():
    passenger = row[FEATURES].to_frame().T
    prediction = int(model.predict(passenger)[0])
    label = "Survived" if prediction == 1 else "Did not survive"
    print(f"Incoming record {index + 1}: {label}")
    time.sleep(1)
