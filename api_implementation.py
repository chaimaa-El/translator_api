# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:22:33 2023

@author: LENOVO
"""

import json
import requests


url = 'http://127.0.0.1:8000/predict'

input_data_for_model = {
    
    
    'text': 'أنا سعيد اليوم ',
    'source_lg' : 'ar',
    'target_language' :'en'
    }

#print(input_data_for_model.text)

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)
print(response.text)
