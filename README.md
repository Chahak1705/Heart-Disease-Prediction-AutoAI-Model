# Heart-Disease-Prediction-AutoAI-
import requests

API_KEY = "<Your Api Key"

# Get token
token_response = requests.post(
    'https://iam.cloud.ibm.com/identity/token',
    data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
)
mltoken = token_response.json()["access_token"]

# Headers
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# Payload
payload_scoring = {
    "input_data": [
        {
            "fields": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                       "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
            "values": [
                [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3],
                [53, 1, 0, 140, 203, 1, 0, 155, 1, 3.1, 0, 0, 3]
            ]
        }
    ]
}

# POST to model
response_scoring = requests.post(
    'https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/1b5f744a-3a01-49ea-a023-b4951f5329d2/predictions?version=2021-05-01',
    json=payload_scoring,
    headers=header
)

print("Scoring response")
try:
    print(response_scoring.json())
except ValueError:
    print(response_scoring.text)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
