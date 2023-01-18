# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:50:12 2023

@author: LENOVO
"""

import requests

data = {"text": "Hello, how are you?", "source_lg": "en", "target_language": "fr"}

response = requests.post("http://127.0.0.1:8000/predict", json=data)

print(response)

print(response.json())