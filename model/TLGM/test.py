import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import Qwen2ForCausalLM

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
# This will make only the GPU 6 visible to the process.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)  # 这里的 0 是 GPU 的索引号

LOCAL_MODEL_PATH = "./Qwen2.5_0.5B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2ForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,  # 使用本地路径
    # device_map="auto",
    device_map={'':torch.cuda.current_device()},
    # device_map="cuda:6",
    trust_remote_code=False,  # 本地加载时建议设置为False，以避免执行远程代码
    quantization_config=bnb_config,
).to('cuda:0')

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # target_modules=["query_key_value"],
    bias="none",
    task_type= "CAUSAL_LM"
)

model = get_peft_model(prepare_model_for_kbit_training(model), config)

prompt = "<human>: What equipment do I need for rock climbing?  \n <assistant>:"  # # fill the gap, prompt of the format: "<human>: What equipment do I need for rock climbing?  \n <assistant>: ", with an empty response from the assistant
print(prompt)


generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

device = torch.cuda.current_device()

encoding = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

data = load_dataset("./helpful_instructions", trust_remote_code=True)
pd.DataFrame(data["train"])

def generate_prompt(data_point):
    full_prompt = ""
    for sentence in data_point:
        full_prompt += "<human>: " + sentence[0] + "  \n <assistant>: " + sentence[1] + "\n"
    return full_prompt # fill the gap, transform the data into prompts of the format: "<human>: question?  \n <assistant>: response" (DONE)

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt

data = data["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)

import os
import torch

import os

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 提醒用户重启内核
print("Please restart the kernel to apply changes.")


# 确保 PyTorch 看到的是正确的 GPU
print("Available GPUs:", torch.cuda.device_count())

OUTPUT_DIR = "experiments"
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=OUTPUT_DIR,
    max_steps=200,   # try more steps if you can
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="tensorboard",
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()


device = "cuda:0"

encoding = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))