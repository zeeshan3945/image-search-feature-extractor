# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import json_tricks as json


path = '/Users/zeeshan/Documents/Qdrant-App/coin_images/1695.jpg'
image_file = open(path, 'rb')
image_data = base64.b64encode(image_file.read()).decode('utf-8')

res = requests.post('http://localhost:8000/', json = {"prompt" : image_data})

print(res.json())
