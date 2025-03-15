import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


import wandb

# Load LLaMA 3 model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})

# Freeze all parameters in the original model
for param in model.parameters():
    param.requires_grad = False  # Freeze everything

class ModifiedLlama(nn.Module):
    def __init__(self, original_model, hidden_size=4096):
        super().__init__()
        self.original_model = original_model
        self.extra_fc = nn.Linear(hidden_size, hidden_size)  # New FC layer

        # Create a separate copy of lm_head to avoid weight tying issues
        self.lm_head = nn.Linear(hidden_size, original_model.lm_head.out_features, bias=False)
        self.lm_head.weight.data = original_model.lm_head.weight.clone()  # Copy weights

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.no_grad():  # Freeze original model
            outputs = self.original_model.model(input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state  # Get hidden states
        transformed_hidden_states = self.extra_fc(hidden_states)  # Pass through FC
        logits = self.lm_head(transformed_hidden_states)  # Pass through new lm_head

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}



# Replace original model with modified model
modified_model = ModifiedLlama(model)

class ModifiedLlama(nn.Module):
    def __init__(self, original_model, hidden_size=4096):
        super().__init__()
        self.original_model = original_model
        # Initialize extra_fc with bfloat16 dtype
        self.extra_fc = nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16)  # New FC layer

        # create a separate copy of lm_head to avoid weight tying issues
        self.lm_head = nn.Linear(hidden_size, original_model.lm_head.out_features, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight.data = original_model.lm_head.weight.clone().type(torch.bfloat16)  # Copy weights and cast to bfloat16

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.no_grad():  # Freeze original model
            outputs = self.original_model.model(input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state  # Get hidden states
        transformed_hidden_states = self.extra_fc(hidden_states)  # Pass through FC
        logits = self.lm_head(transformed_hidden_states)  # Pass through new lm_head

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

        # Replace original model with modified model
modified_model = ModifiedLlama(model)

class ModifiedLlama(nn.Module):
    def __init__(self, original_model, hidden_size=4096):
        super().__init__()
        self.original_model = original_model
        # Initialize extra_fc with bfloat16 dtype
        self.extra_fc = nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16)  # New FC layer

        # Create a separate copy of lm_head to avoid weight tying issues
        self.lm_head = nn.Linear(hidden_size, original_model.lm_head.out_features, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight.data = original_model.lm_head.weight.clone().type(torch.bfloat16)  # Copy weights and cast to bfloat16

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.no_grad():  # Freeze original model
            outputs = self.original_model.model(input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state  # Get hidden states
        # Cast hidden_states to bfloat16 to match the dtype of extra_fc
        transformed_hidden_states = self.extra_fc(hidden_states.type(torch.bfloat16))  # Pass through FC
        logits = self.lm_head(transformed_hidden_states)  # Pass through new lm_head

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# Replace original model with modified model
modified_model = ModifiedLlama(model)

# Ensure only the extra FC layer is trainable
for param in modified_model.parameters():
    param.requires_grad = False  # Freeze everything

for param in modified_model.extra_fc.parameters():
    param.requires_grad = True  # Unfreeze only the extra FC layer

model

modified_model

# Check trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total number of trainable parameters:", count_parameters(modified_model))

from datasets import load_dataset
from transformers import TrainingArguments

# Load dataset (use train split instead of test)
dataset = load_dataset("HiTZ/EusTrivia", split="test")

# Split dataset into training (90%) and validation (10%)
dataset = dataset.train_test_split(test_size=0.1, seed=42)  # Ensures reproducibility
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Format the dataset for finetuning

def format_eus_trivia(example):
    question = example["question"]
    candidates = example["candidates"]
    answer = example["answer"]

    # Format options
    candidates_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(candidates)])

    # Construct input-output pair
    formatted_example = {
        "prompt": f"Basque Trivia Question:\n{question}\n\n{candidates_text}\n\nAnswer:",
        "response": f"{answer}"
    }
    return formatted_example

#Apply formatting
train_dataset = train_dataset.map(format_eus_trivia)
val_dataset = val_dataset.map(format_eus_trivia)

# Tokenization function
def tokenize_function(examples):
    texts = [p + "\n\n" + r + tokenizer.eos_token for p, r in zip(examples["prompt"], examples["response"])]

    model_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    model_inputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in model_inputs["input_ids"]
    ]

    return model_inputs

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names, num_proc=1)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names, num_proc=1)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

#Define the training arguments
training_args = TrainingArguments(
    output_dir="./llama3-basque",
    num_train_epochs=5,
    per_device_train_batch_size=1, # specifies the batch size for training on each device
    gradient_accumulation_steps=8, #controls how many steps the model will
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="wandb",
    weight_decay=0.01,
    label_names=["labels"],
    optim="adamw_hf",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

)

from transformers import Trainer

trainer = Trainer(
    model=modified_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained("llama3/1fc/basque")
tokenizer.save_pretrained("llama3/1fc/basque")

import torch
torch.cuda.empty_cache()

!pip install git+https://github.com/EleutherAI/lm-evaluation-harness

datseteval = load_dataset("HiTZ/BertaQA", 'en', split="test")

# select tasks
tasks_selected=(
    "bertaqa_eu"
    "bertaqa_en"
    #"bertaqa_en_mt"
)

!lm_eval --model hf \
    --model_args pretrained=./llama3/1fc/basque \
    --tasks bertaqa_en \
    --device cuda:0 \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path ./results/llama3fc/finetuned \
    --log_samples

!lm_eval --model hf \
    --model_args pretrained=./llama3/1fc/basque \
    --tasks bertaqa_eu \
    --device cuda:0 \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path ./results/llama3fc/finetuned \
    --log_samples

