import requests

url = "http://127.0.0.1:8000/predict"
passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}
response = requests.post(url, json=passenger, timeout=10)
print("Status code:", response.status_code)
print("Response:", response.json())
