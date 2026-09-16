import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import numpy as np
import argparse
import torch
import json

# Load BERT model and tokenizer
def load_bert_model(device):
    tokenizer = AutoTokenizer.from_pretrained("/home/qian/chenshangheng/graph360/topic/data/script/bert-base-chinese")
    model = AutoModelForMaskedLM.from_pretrained("/home/qian/chenshangheng/graph360/topic/data/script/bert-base-chinese")
    
    # Move model and tokenizer to the given device
    tokenizer = tokenizer
    model = model.to(device)

    return tokenizer, model

# Extract textual features from BERT
def bert_textual_feature_extraction(tokenizer, model, text, device):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Move inputs to the device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract [CLS] token embedding (the first token in the sequence)
    cls_embedding = outputs.logits[:, 0, :].cpu().numpy()  # shape: (batch_size, hidden_size)
    
    return cls_embedding.flatten()  # Flattening to a 1D array

if __name__ == '__main__':
    # Argument parser for device selection
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1', help='Device for training')
    args = parser.parse_args()
    device = torch.device(args.device)
    target_type_list = ['content', 'title', 'desc', 'comment', 'topic']
    # target_type = '内容'  # Target text field
    tokenizer, model = load_bert_model(device)
    for target_type in target_type_list:
    # Load dataset from CSV
        source_path = f'/home/qian/chenshangheng/graph360/topic/data/{target_type}.csv'
        

        text_embs = []  # List to store embeddings
        
        # Read CSV file
        df = pd.read_csv(source_path)
        
        # Ensure 'text' column exists
        if 'text' not in df.columns:
            raise ValueError(f"The CSV file {source_path} does not contain a 'text' column.")
        
        # Process each row in the dataset
        for text in tqdm(df['text']):
            # Ensure text is not empty
            if pd.isna(text) or text.strip() == "":
                text = " "  # Replace empty text with a space
            text_emb = bert_textual_feature_extraction(tokenizer, model, text, device)
            text_embs.append(text_emb)

        # Save embeddings as a numpy array
        np.save(target_type + '.npy', np.array(text_embs))
        print(target_type + ' Save')
