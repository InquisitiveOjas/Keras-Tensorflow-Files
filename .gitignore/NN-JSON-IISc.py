# -*- coding: utf-8 -*-
"""
Created on Mon May 13 08:39:01 2019

@author: ojasr
"""

from keras.models import model_from_json
import json




with open("Downloads\AlexNet_Sept_19.json", "r") as r:
   data = json.load(r)

with open ("Downloads\AlexNet_Sept_19.json" , "r") as f:
    model2 = model_from_json(f.read())
print (model2.summary())


json_file = open("Downloads\AlexNet_Sept_19.json", "r")
loaded_model_json = json_file.read()
json_file.close()
