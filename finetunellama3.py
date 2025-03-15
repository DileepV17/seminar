# import dependencies
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

#model = AutoModelForCausalLM.from_pretrained(
    #model_name, torch_dtype=torch.float16, device_map="auto"
#)

#loading the ,odel in 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})

import peft
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Apply LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("HiTZ/EusTrivia", split="test")

# Format the dataset for finetuning

def format_eus_trivia(example):
    question = example["question"] # question
    candidates = example["candidates"] # all the candidates
    answer = example["answer"] # indicates the index of the answer

    # Format options
    candidates_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(candidates)])

    # Construct input-output pair
    formatted_example = {
        "prompt": f"Basque Trivia Question:\n{question}\n\n{candidates_text}\n\nAnswer:",
        "response": candidates[answer] # converts the index of the candidates to actual text answer.
    }
    return formatted_example

# Format dataset
formatted_dataset = dataset.map(format_eus_trivia)

def tokenize_function(examples):
    # Format prompt and response for better instruction following
    texts = [p + "\n\n" + r + tokenizer.eos_token for p, r in zip(examples["prompt"], examples["response"])]

    # Tokenize with padding and truncation
    model_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # Set labels: ignore padding tokens (-100)
    model_inputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in model_inputs["input_ids"]
    ]

    return model_inputs

# Tokenize dataset
tokenized_datasets = formatted_dataset.map(tokenize_function, batched=True,
                                           remove_columns=formatted_dataset.column_names,
                                           num_proc=4)

tokenized_datasets.set_format("torch")  # Ensure PyTorch compatibility


#Define the training arguments
training_args = TrainingArguments(
    output_dir="./llama3-basque",
    num_train_epochs=6, #10
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4, # 3e-5 / 5e-5
    optim="paged_adamw_32bit",
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    report_to="none",
    weight_decay=0.01
)

from transformers import Trainer

#initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets


)

import torch
torch.cuda.empty_cache()

#start the training

trainer.train()

save_path ="/dss/dsshome1/07/ra47fey2/finetuned_llama3"



model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

