
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class Classifier(nn.Module):
    def __init__(self,n_classes):
        super(Classifier,self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self,input_ids, attention_mask):
        _,pooled_output = self.bert(input_ids, attention_mask= attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

class IMDBDataset(Dataset):
    
    def __init__(self, reviews, sentiments, tokenizer, max_len):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len == max_len
    
    def __len__(self):
    return len(self.reviews)
  
    def __getitem__(self, item):
        review = str(self.reviews[item])
        sentiment = self.sentiments[item]

        encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        )

        return {
        'review': review,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'sentiments': torch.tensor(sentiment, dtype=torch.long)
        }





def data_exploration_visualization():
    df = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
    print(df.head())
    print(df.shape)



