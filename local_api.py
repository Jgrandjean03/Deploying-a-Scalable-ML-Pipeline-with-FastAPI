import requests

BASE_URL = "http://127.0.0.1:8000"
POST_PATH = "/data/"
REQUEST_TIMEOUT_SECONDS = 10

SAMPLE_RECORD = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# send a GET using the base URL
r = requests.get(BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)

# print the status code
print(f"Status Code: {r.status_code}")
# print the welcome message
print(f"Result: {r.json()['message']}")

# send a POST using the sample record
r = requests.post(
    BASE_URL + POST_PATH, json=SAMPLE_RECORD, timeout=REQUEST_TIMEOUT_SECONDS
)

# print the status code
print(f"Status Code: {r.status_code}")
# print the result
print(f"Result: {r.json()['result']}")
