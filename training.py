import torch
import torch.nn as nn
import numpy as np
import os
import evaluate
from dotenv import load_dotenv
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from preprocess import preprocess_training
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.training_args import OptimizerNames
from datasets import DatasetDict, Dataset

load_dotenv('.env')
HF_TOKEN = os.getenv("HF_TOKEN")


def get_fine_tuning_trainer_args(output_path):

    return TrainingArguments(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=1,
        save_steps=20,
        eval_steps=20,
        logging_steps=10,
        learning_rate=1e-5,
        warmup_ratio=0.1,
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


def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': sum([1 for i, j in zip(predictions, labels) if i == j])/len(labels)}


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def fine_tuning():

    train_dataset, test_dataset = preprocess_training()

    dataset_train = Dataset.from_list(train_dataset)
    dataset_test = Dataset.from_list(test_dataset)

    dataset = DatasetDict({
        'train': dataset_train,
        'eval': dataset_test,
    })

    fine_tune_args = get_fine_tuning_trainer_args("results/")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                                                 quantization_config=quantization_config,
                                                 num_labels=2,
                                                 cache_dir='model/')

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    lora_config = LoraConfig(
        r=16,  # attention heads
        lora_alpha=32,  # alpha scaling
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"  # set this for CLM or Seq2Seq
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.add_adapter(peft_config=lora_config, adapter_name="adapter_1")
    model.set_adapter("adapter_1")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir='model/')

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print_trainable_parameters(model)

    def data_preprocesing(row):
        return tokenizer(row['input'], truncation=True, max_length=5000)

    tokenized_data = dataset.map(data_preprocesing, batched=True,
                                 remove_columns=['input'])
    tokenized_data.set_format("torch")


    fine_tune_trainer=Trainer(
        model=model,
        args=fine_tune_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['eval'],
        compute_metrics=compute_metrics
    )

    train_results = fine_tune_trainer.train()

    fine_tune_trainer.save_model(output_dir='results/models')
    fine_tune_trainer.log_metrics("train", train_results.metrics)
    fine_tune_trainer.save_metrics("train", train_results.metrics)
    fine_tune_trainer.save_state()


if __name__ == '__main__':
    fine_tuning()