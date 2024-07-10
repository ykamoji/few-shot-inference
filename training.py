import torch
import numpy as np
import os
import evaluate
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from preprocess import preprocess_training
from transformers import Trainer, TrainingArguments
from transformers.training_args import OptimizerNames

load_dotenv('.env')
HF_TOKEN = os.getenv("HF_TOKEN")

def get_fine_tuning_trainer_args(output_path):

    return TrainingArguments(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=,
        per_device_eval_batch_size=,
        evaluation_strategy="steps",
        num_train_epochs=,
        save_steps=20,
        eval_steps=20,
        logging_steps=10,
        learning_rate=,
        warmup_ratio=0.1,
        weight_decay=,
        save_total_limit=2,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        optim=OptimizerNames.ADAMW_HF,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=42,
        gradient_accumulation_steps=1,
    )

def fine_tuning():

    dataset = preprocess_training()

    fine_tune_args = get_fine_tuning_trainer_args("")

    fine_tune_trainer = Trainer(
        model=,
        args=fine_tune_args,
        data_collator=,
        train_dataset=,
        eval_dataset=,
    )