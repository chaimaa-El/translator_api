# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:15:26 2023

@author: LENOVO
"""

#from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFMarianMTModel, MarianTokenizer
import json


#app = FastAPI()

class model_input(BaseModel):
    
    text : str
    source_lg : str
    target_language : str
          
        
# loading the saved model
model1 = TFMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ar')
tokenizer1 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ar')        
model2 = TFMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
tokenizer2 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
"""
model3 = TFMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-ar')
tokenizer3 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-ar')
model4 = TFMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
tokenizer4 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
model5 = TFMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ar-en')
tokenizer5 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ar-en')
model6 = TFMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ar-fr')
tokenizer6 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ar-fr')
"""

#@app.post('/predict')
def translate(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    source_lg = input_dictionary["source_lg"]
    target_language = input_dictionary["target_language"]
    text = input_dictionary["text"]
    
    print(text,source_lg,target_language)

 

    """source_lg = input_parameters.source_lg
    target_language = input_parameters.target_language
    text = input_parameters.text"""
    
    if source_lg == "en" and target_language == "ar":
        input_ids = tokenizer1.encode(text, return_tensors="tf")
        outputs = model1.generate(input_ids)
        translated_text = tokenizer1.decode(outputs[0], skip_special_tokens=True)
        return("Translated text in "+target_language+": " + translated_text)


    elif source_lg == "en" and target_language == "fr":
        input_ids = tokenizer2.encode(text, return_tensors="tf")
        outputs = model2.generate(input_ids)
        translated_text = tokenizer2.decode(outputs[0], skip_special_tokens=True)
        print("Translated text in "+target_language+": " + translated_text)
        return("Translated text in "+target_language+": " + translated_text)
        """
   elif source_lg == "fr" and target_language == "ar":
        input_ids = tokenizer3.encode(text, return_tensors="pt")
        outputs = model3.generate(input_ids)
        translated_text = tokenizer3.decode(outputs[0], skip_special_tokens=True)
        return("Translated text in "+target_language+": " + translated_text)
    elif source_lg == "fr" and target_language == "en":
        input_ids = tokenizer4.encode(text, return_tensors="pt")
        outputs = model4.generate(input_ids)
        translated_text = tokenizer4.decode(outputs[0], skip_special_tokens=True)
        return("Translated text in "+target_language+": " + translated_text)
    elif source_lg == "ar" and target_language == "en":
        input_ids = tokenizer5.encode(text, return_tensors="pt")
        outputs = model5.generate(input_ids)
        translated_text = tokenizer5.decode(outputs[0], skip_special_tokens=True)
        return("Translated text in "+target_language+": " + translated_text) 
    """
    
text='PyTorch Hub is a pre-trained model repository designed to facilitate research reproducibility.'
#translate(text,"en","ar")
input_data = {'text': text, 'source_lg': 'en', 'target_language': 'fr'}
input_object = model_input(**input_data)
#input_object = model_input(text, source_lg="en", target_language="fr")

translate(input_object)