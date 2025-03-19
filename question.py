# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 22:54:08 2025

@author: Onur
"""



# from transformers import AutoTokenizer, TFAutoModelForCausalLM

# # Model ve tokenizer yükle, padding'i sola ayarla
# model_name = "microsoft/DialoGPT-medium"
# tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")  # Padding sola alındı

# model = TFAutoModelForCausalLM.from_pretrained(model_name)

# # Padding token'ı ayarla
# tokenizer.pad_token = tokenizer.eos_token

# # Sohbet döngüsü
# while True:
#     soru = input("Sorunuzu girin (çıkmak için 'çık'): ")
#     if soru == "çık":
#         break
    
#     # Soruyu tokenize et ve modele ver
#     inputs = tokenizer.encode(soru + tokenizer.eos_token, return_tensors="tf")
#     outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    
#     # Cevabı decode et, soruyu hariç tut
#     cevap = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     cevap = cevap[len(soru):].strip()  # Soruyu cevaptan çıkar
#     print("Cevap:", cevap)
    
    
    
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from fastapi import FastAPI

# FastAPI uygulamasını başlat
app = FastAPI()

# Model ve tokenizer'ı yükle
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = TFAutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# API endpoint'i tanımla
@app.get("/soru")
async def get_cevap(text: str):
    # Soruyu tokenize et ve modele ver
    inputs = tokenizer.encode(text + tokenizer.eos_token, return_tensors="tf")
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    
    # Cevabı decode et, soruyu hariç tut
    cevap = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cevap = cevap[len(text):].strip()
    return {"cevap": cevap}


