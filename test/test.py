from customer import customer
import requests

json = customer
url = "http://127.0.0.1:8000/predict"

response = requests.post(url, json = json)
print(response.json())

