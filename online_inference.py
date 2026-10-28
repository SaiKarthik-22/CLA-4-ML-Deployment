import pandas as pd
import joblib

model = joblib.load("model.pkl")
records = pd.read_csv("unseen_records.csv")
row = records.iloc[[0]][["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
prediction = int(model.predict(row)[0])

print("Online inference - one unseen passenger")
print(row.to_string(index=False))
print("Prediction:", "Survived" if prediction == 1 else "Did not survive")
