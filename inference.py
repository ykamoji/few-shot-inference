import torch
import numpy as np
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from preprocess import preprocess
import re
from tqdm import tqdm

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

for key in dataset.keys():
    print(f"Processing criteria {key+1}")

    failed_cases = 0
    accuracy = 0
    
    pbar = tqdm(enumerate(dataset[key]))
    for idx, item in pbar:
        prompt_input = item["prompt"]
        prompt_len = len(prompt_input)
        input_tokens = tokenizer(prompt_input, return_tensors="pt")

        model.to('cuda')
        input_tokens = input_tokens.to('cuda')
        result_sample = model.generate(**input_tokens, max_new_tokens= 200, do_sample=True,top_k=1, temperature = 0.5)
        # print("result_sample: ", result_sample[0])

        # print(f"Response {idx}:")
        response = tokenizer.decode(result_sample[0])
        # print(response)
        # print(f"--")
        output = response[prompt_len:prompt_len+10]
        # print(f"{output}")
        search = re.findall(r'\d+',output)
        if len(search) > 0:
            target = 1 if item["target"] == "Satisfactory" else 0
            decision = search[0]
            prediction = int(decision) == target
            # print(f"Predicted {decision} Actual {target}")
        else:
            failed_cases += 1
            prediction = False

            # print(prediction)
        if prediction:
            accuracy += 1

        acc_per = accuracy * 100 / (idx + 1)

        pbar.set_description(f"Accuracy {acc_per:.2f} %")

        # print("\n\n")

    accuracy = accuracy * 100 / len(dataset[key])

    print(f"Total failed cases {failed_cases}")
    print(f"Accuracy = {accuracy:.2f} %\n\n")
