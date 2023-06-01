
# import data preprocessing libraries

import os
import re


import pyarrow
import pandas as pd
import numpy as np

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel, AutoConfig, 
    AutoTokenizer
)

from sentence_transformers import util


def to_string(series) -> str :
    sentence = ''
    for word in series:
        sentence += word+' [SEP] '
    sentence = sentence.strip()
    return sentence

class PreprocessPipeLine:
    
    def __call__(self, df:pd.DataFrame):
        """
        drop redundant features not necesaary for retrieval system
        `data_df.drop(list of columns to drop, axis=1)`
        """
        df = df.drop(['rating','no_of_reviews',
                      'drug_link','medical_condition_url',
                      'medical_condition_description'],
                     axis=1)
        # replace null values with the string unknown
        df = df.fillna('unknown')
        
        df['related_drugs'] = df['related_drugs'].apply(lambda z: self.remove_url_char(z))
        
        # explode `side_effects` column
        df['side_effects'] = df['side_effects'].apply(lambda z: z.split('.'))
        df = df.explode('side_effects', ignore_index=True)
        
        # drop rows with empty side_effects
        df['string_length'] = df['side_effects'].apply(lambda z: len(z))
        df = df[df['string_length']>0]
        df.index = [i for i in range(len(df))]
        df = df.drop('string_length', axis=1)
        return df
    
    def remove_url_char(self, feature):
        # clean related drugs:
            #remove unwanted url links from 
            #remove characters '|' and spaces
        url_cleaner = re.compile(r":|https://\S+|www\.\S+")
        feature = url_cleaner.sub(r'', feature)
        feature = feature.strip().replace(r'  | ', ', ')
        return feature




class DrugsInformation(torch.utils.data.Dataset):
    def __init__(self, drugs, tokenizer, preprocess=None):
        self.drugs = drugs
        self.preprocess = preprocess
        
    def __getitem__(self, idx:int) -> str:
        if isinstance(self.drugs, pd.DataFrame) and self.preprocess is not None:
            drug_info = self.preprocess(self.drugs).loc[idx]
            #drug_info = self.drugs.loc[idx]
        else:
            drug_info = self.drugs[idx]           
        drug_info = to_string(drug_info)   
        
        return drug_info

    def __len__(self):
        return len(self.preprocess(self.drugs))
