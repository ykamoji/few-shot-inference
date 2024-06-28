import torch
import numpy as np
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from preprocess import preprocess

load_dotenv('.env')
HF_TOKEN = os.getenv("HF_TOKEN")

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,
#     llm_int8_has_fp16_weight=False
# )

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir='model/')
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir='model/') #quantization_config=quantization_config

dataset = preprocess()

input_tokens = tokenizer(dataset[0]["prompt"], return_tensors="pt")

model.to('mps')
input_tokens = input_tokens.to('mps')
result_sample = model.generate(**input_tokens, max_new_tokens= 200, do_sample=False,top_k=1 , temperature = 0.5, skip_special_tokens=True)
# print("result_sample: ", result_sample[0])

print("Response:")
print(tokenizer.decode(result_sample[0]))