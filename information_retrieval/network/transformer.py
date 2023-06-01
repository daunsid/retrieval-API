from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TFAutoModel
    
)

import torch.nn as nn

import os
import typing

from information_retrieval import config

model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(nn.Module):
    def __init__(self, model_name_path:str):
        super(Encoder, self).__init__()
        if os.path.exists(model_name_path+'/config.json'):
            self.config = AutoConfig.from_pretrained(model_name_path)
            self.model = AutoModel.from_pretrained(model_name_path, from_pt=True, config=self.config)

        else:
            model_name=config['MODEL_CONFIG']['model_name']
            self.model = AutoModel.from_pretrained(model_name)
            self.model.save_pretrained(model_name_path)
        

    def forward(self, **encoded_inputs):
        out = self.model(**encoded_inputs)
        
        return out