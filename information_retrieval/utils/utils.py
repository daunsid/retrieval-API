import torch
import pickle
import os

import torch.nn as nn


from transformers import (
    AutoTokenizer
)

from ..network.transformer import Encoder
from information_retrieval import config, device
#Mean Pooling - Take attention mask into account for correct averaging

class MeanPooling(nn.Module) :
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, model_output, attention_mask) -> torch.tensor :

        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask =  torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
def get_embeddings(encoded_inputs):
    encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
    with torch.no_grad():
        model = Encoder(config['PATHS']["MODEL_PATH"]).to(device)
        model_output = model(**encoded_inputs)
        
    pooler = MeanPooling()
    embeddings = pooler(model_output, encoded_inputs['attention_mask'])
    embeddings = nn.functional.normalize(embeddings)
    return embeddings

def load_embeddings(embd_path:str):
    with open(embd_path, 'rb') as pkfile:
        emb = pickle.load(pkfile)
        pkfile.close()

def get_tokenizer(texts, model_name_path):
    if os.path.exists(model_name_path+'/config.json'):
        tokenizer = AutoTokenizer.from_pretrained(model_name_path)
        
    else:
        model_name=config['MODEL_CONFIG']['model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_name_path)

    encoded_texts = tokenizer(
        texts,add_special_tokens=True,padding='max_length',
        max_length=int(config['MODEL_CONFIG']["max_length"]),
        truncation=True,
        return_tensors='pt',
        #return_attention_mask=True
    )                          
    return encoded_texts