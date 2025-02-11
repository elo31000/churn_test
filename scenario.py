import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Age": 30,
    "Years": 2,
    "Num_Sites": 3,
    "Account_Manager": 0
}

response = requests.post(url, json=data)

print(response.json())