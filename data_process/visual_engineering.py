import pandas as pd
import torch
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
import requests
import os
import argparse


def load_vit_model(device):

    processor = ViTImageProcessor.from_pretrained(
        r'/home/qian/chenshangheng/MMRA-main/data/model/vit-base-patch16-224-in21k')

    model = ViTModel.from_pretrained(
        r'/home/qian/chenshangheng/MMRA-main/data/model/vit-base-patch16-224-in21k').to(device)

    return processor, model


def vit_visual_feature_extraction(processor, model, image_path, device):

    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)

    cls_output = outputs.last_hidden_state[:, 0, :]

    return (cls_output[0]).tolist()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:4', help='Device for training')
    args = parser.parse_args()

    device = torch.device(args.device)

    processor, model = load_vit_model(device)

    visual_embs = []
    for i in tqdm(range(1, 96936+1)):
        visual_emb = []

        for j in range(8):
            image_path = f'/home/qian/chenshangheng/graph360/topic/data/frame/{i}_{j}.jpg'

            try:  # Check if file exists
                image_emb = vit_visual_feature_extraction(processor, model, image_path, device)
                visual_emb.append(image_emb)
            except:
                continue

        if visual_emb:
            # Average the valid image embeddings
            averaged_emb = np.mean(visual_emb, axis=0)
            visual_embs.append(averaged_emb)
        else:
            # If no valid images, append a zero vector of size 768
            zero_emb = np.zeros(768)  # 768-dimensional zero vector
            visual_embs.append(zero_emb)

    np.save('../visual_feature_embedding_cls.npy', np.array(visual_embs))