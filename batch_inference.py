import pandas as pd
import joblib

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
model = joblib.load("model.pkl")
records = pd.read_csv("unseen_records.csv").head(20).copy()
records["Predicted_Survived"] = model.predict(records[FEATURES]).astype(int)
records["Prediction_Label"] = records["Predicted_Survived"].map(
    {1: "Survived", 0: "Did not survive"}
)
records.to_csv("batch_predictions.csv", index=False)
print(records.to_string(index=False))
print("\nSaved 20-record results to batch_predictions.csv")
