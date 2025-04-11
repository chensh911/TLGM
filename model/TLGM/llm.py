import torch
import torch.nn as nn
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union

from transformers import PreTrainedModel as Qwen2PreTrainedModel
from transformers import BertModel as Qwen2Model 

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward

from transformers import logging
from transformers.models.qwen2.modeling_qwen2 import *

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

logger = logging.get_logger(__name__)


class Qwen2WithGraphForRegression(Qwen2ForSequenceClassification):
    def __init__(self, config, quantization_config=None):
        super().__init__(config)
        
        # Add the graph projector to match graph features with hidden size
        if quantization_config is not None:
            self.config.quantization_config = quantization_config
        config.graph_feature_size = 2048 // 4
        self.graph_projector = nn.Linear(config.graph_feature_size, config.hidden_size, dtype=torch.float32)
        self.graph_projector.reset_parameters()
        
        # Linear layer for regression output
        self.regressor = nn.Linear(config.hidden_size, 1)  # Output a single regression value
        
        # Loss function for regression
        self.loss_fct = nn.SmoothL1Loss()

    def freeze_model(self):
        '''freeze pretrained parameters, unfreeze additional parameters'''
        for para in self.parameters():
            para.requires_grad = False
        for para in self.graph_projector.parameters():
            if para.dtype.is_floating_point:
                para.requires_grad = True
        for para in self.regressor.parameters():
            if para.dtype.is_floating_point:
                para.requires_grad = True
    
    def unfreeze_model(self):
        '''freeze pretrained parameters, unfreeze additional parameters'''
        for para in self.parameters():
            if para.dtype.is_floating_point:
                para.requires_grad = True

    def forward(
        self,
        graph_feature: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        # If inputs_embeds is not provided, generate it from input_ids
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        # If graph_feature is provided, project it and add it to the embeddings
        if graph_feature is not None:
            graph_feature = graph_feature.view(graph_feature.shape[0], 4, -1)
            graph_embeds = self.graph_projector(graph_feature)
            # graph_embeds = graph_embeds.unsqueeze(1)

            graph_mask = (input_ids == self.config.quantization_config.graph_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(graph_mask, graph_embeds)

        # Now run the transformer model
        transformer_outputs = self.model(
            input_ids=None,  # We are using inputs_embeds here
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]  # The hidden states from the transformer

        # Use the hidden state of the last token for regression prediction
        graph_token_mask = (input_ids == self.config.quantization_config.graph_token_id)
        # last_token_pos = graph_token_mask.int().argmax(dim=1)

        last_token_pos = attention_mask.sum(dim=-1) - 1
        hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_token_pos]

        # Regression prediction
        pred = self.regressor(hidden_states)

        # Compute the regression loss (if labels are provided)
        loss = None
        if labels is not None:
            loss = self.loss_fct(pred.squeeze(-1), labels.float())  # Convert labels to float for regression
        
        # Return the results
        if not return_dict:
            return (loss, pred) if loss is not None else (pred,)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pred,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

def initialize_model(model_name, label, device, lora=True, freeze=False, prompt_count=0):
    # Configure BitsAndBytes for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the tokenizer and set special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set
    tokenizer.add_special_tokens({
        "additional_special_tokens": ['<|graph_start|>', '<|graph_end|>', '<|graph_pad|>']
    }, replace_additional_special_tokens=False)
    
    # Define a special graph token
    tokenizer.graph_token = '<|graph_pad|>'
    bnb_config.graph_token_id = tokenizer.convert_tokens_to_ids(tokenizer.graph_token)

    # Load the model with specified configurations and quantization
    model = Qwen2WithGraphForRegression.from_pretrained(
        model_name,  # Load model from local path
        # device_map={'': torch.cuda.current_device()},  # Assign model to current GPU device
        trust_remote_code=False,  # Disable remote code execution
        quantization_config=bnb_config,  # Apply quantization configuration
    )#.to(device)  # Move the model to the specified device
    

    # Construct the prompt for regression task
    
    prompt1 = f"Given a graph token, predict the {label} popularity metric based on the historical data linked to this graph token. Output the prediction along with a unique token for regression analysis. Graph token: <|graph_pad|> <|graph_pad|> <|graph_pad|> <|graph_pad|>, Popularity: "
    
    prompt2 = f"Using the historical data associated with this graph token, predict the {label} popularity metric. Output the predicted value along with a unique token for regression analysis. Graph token: <|graph_pad|> <|graph_pad|> <|graph_pad|> <|graph_pad|>, Popularity: "
    prompt3 = f"Given a graph token, forecast the {label} popularity metric based on the linked historical data. Provide the prediction and a unique token for regression analysis. Graph token: <|graph_pad|> <|graph_pad|> <|graph_pad|> <|graph_pad|>, Popularity: "
    prompt4 = f"Based on the historical data associated with this graph token, predict the {label} popularity metric. The prediction should include a unique token for regression analysis. Graph token: <|graph_pad|> <|graph_pad|> <|graph_pad|> <|graph_pad|>, Popularity: "
    prompt5 = f"Predict the {label} popularity metric for the given graph token, using the historical data linked to it. Output the prediction along with a unique token for regression analysis. Graph token: <|graph_pad|> <|graph_pad|> <|graph_pad|> <|graph_pad|>, Popularity: "
    prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5]



    # Tokenize the prompt
    encoding = tokenizer(prompt_list[prompt_count], return_tensors="pt").to(device)
    if lora:
        # Configure LoRA settings for low-rank adaptation
        config = LoraConfig(
            r=16,               # Low-rank adaptation dimension
            lora_alpha=32,      # Scaling factor for LoRA layers
            lora_dropout=0.05,  # Dropout rate for LoRA layers
            bias="none",        # Bias term setting
            task_type="CAUSAL_LM"  # Task type for causal language modeling SEQ_CLS
        )

        # Apply LoRA and prepare the model for k-bit training
        model = get_peft_model(prepare_model_for_kbit_training(model), config)
    else:
        # model.freeze_lora_model()
        model = prepare_model_for_kbit_training(model)
        model.unfreeze_model()
    if freeze:
        model.freeze_model()


    return model, tokenizer, encoding




