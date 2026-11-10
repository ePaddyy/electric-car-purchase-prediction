# from customer import customer
# import requests

# json = customer
# url = "http://127.0.0.1:8000/predict"

# response = requests.post(url, json = json)
# print(response.json())

import requests
from customer.customer import customer

url = "http://localhost:8000/predict"
response = requests.post(url, json=customer)
print(response.status_code)
print(response.json())