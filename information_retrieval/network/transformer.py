from transformers import (
    AutoTokenizer, AutoModel, AutoConfig
)
import torch.nn as nn

import os
import typing

from information_retrieval import config

model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(nn.Module):
    def __init__(self, model_name_path:str):
        super(Encoder, self).__init__()
        self.config = AutoConfig.from_pretrained(config['model_name_path'])
        self.model = AutoModel.from_pretrained(config['model_name_path'], from_tf=True, config=self.config)

    def forward(self, **encoded_inputs):
        out = self.model(**encoded_inputs)
        
        return out