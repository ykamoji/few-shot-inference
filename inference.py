import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import csv
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,DataCollatorWithPadding
from preprocess import preprocess_inference
from datasets import DatasetDict, Dataset
from transformers import BitsAndBytesConfig
import re
from tqdm import tqdm
import warnings
from peft import PeftModel
warnings.filterwarnings("ignore")

load_dotenv('.env')
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir = "model/")
# base_model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir = "model/")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                                             quantization_config=quantization_config,
                                             num_labels=2,
                                             cache_dir='model/')

new_model = "/home/sujeetj-pg/whisper/mist/sequence-cls/results_2/training/checkpoint-635"
model = PeftModel.from_pretrained(base_model, new_model)
device = 'cuda'

test_dataset = preprocess_inference()


def generation_inference():
    for key in test_dataset.keys():

        print(f"Processing criteria {key + 1}")

        failed_cases = 0
        accuracy = 0

        pbar = tqdm(enumerate(test_dataset[key]))
        for idx, item in pbar:
            prompt_input = item["prompt"]
            prompt_len = len(prompt_input)
            input_tokens = tokenizer(prompt_input, return_tensors="pt")

            model.to('cuda')
            input_tokens = input_tokens.to('cuda')
            result_sample = model.generate(**input_tokens, max_new_tokens=200, do_sample=True, top_k=1, temperature=0.5)
            # print("result_sample: ", result_sample[0])

            # print(f"Response {idx}:")
            response = tokenizer.decode(result_sample[0])
            # print(response)
            # print(f"--")
            output = response[prompt_len:prompt_len + 10]
            # print(f"{output}")
            search = re.findall(r'\d+', output)
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

        accuracy = accuracy * 100 / len(test_dataset[key])

        print(f"Total failed cases {failed_cases}")
        print(f"Accuracy = {accuracy:.2f} %\n\n")


def sequence_inference():
    dataset_test = Dataset.from_list(test_dataset)

    dataset = DatasetDict({
        'eval': dataset_test,
    })

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    def data_preprocesing(row):
        return tokenizer(row['input'], truncation=True, max_length=5000)

    tokenized_data = dataset.map(data_preprocesing, batched=True, remove_columns=['input'])
    tokenized_data.set_format("torch")

    collector_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    model.eval()
    model.to(device)

    labels = []
    predictions = []
    csvData = []
    pbar = tqdm(enumerate(tokenized_data['eval']))
    for idx, test_input in pbar:
        tokenized_input = {k: v for k, v in test_input.items() if k != 'labels'}

        test_data = collector_fn(tokenized_input)

        with torch.no_grad():
            output = model(**{k: v.unsqueeze(0).to(device) for k, v in test_data.items()})

        logits = output.logits.cpu().numpy()

        prediction = np.argmax(logits, axis=1)[0]

        labels.append(test_input['labels'].item())
        predictions.append(prediction)

        csvData.append({"Prompt": dataset_test[idx]['input'],
                        "Predicted": "Satisfactory" if prediction == 1 else "Not Satisfactory",
                        "Actual": "Satisfactory" if test_input['labels'] == 1 else "Not Satisfactory"})

    print("Accuracy\t\tBalanced Accuracy\t\tPrecision score\t\tRecall score\t\tF1 score")

    print(f"{accuracy_score(labels, predictions):.2f} %\t\t\t{balanced_accuracy_score(labels, predictions):.2f} %\t\t\t\t"
           f"{precision_score(labels, predictions):.2f} %\t\t\t{recall_score(labels, predictions):.2f} %\t\t\t"
           f"{f1_score(labels, predictions):.2f} %")

    with open("inference_results.csv", "w") as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=["Prompt", "Predicted", "Actual"])
        writer.writeheader()
        for row_data in csvData:
            writer.writerow(row_data)


if __name__ == '__main__':
    sequence_inference()
