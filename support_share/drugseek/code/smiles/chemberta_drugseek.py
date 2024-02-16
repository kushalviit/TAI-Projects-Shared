from transformers import RobertaTokenizer, RobertaModel, AutoModelForMaskedLM, AutoTokenizer
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import BertForSequenceClassification, BertConfig
import pandas as pd




class CBsmiles(nn.Module):
    def __init__(self, main_model,pool_type):
        super(CBsmiles, self).__init__()
        self.main_model = main_model
        self.pool_type = pool_type
    
    def forward(self,input_ids,attention_mask):
        hiddenState, ClsPooled = self.main_model(input_ids = input_ids,attention_mask=attention_mask).values()
        if self.pool_type.lower() == "max":
            embeddings = self.max_pooling(hiddenState, attention_mask)
        elif self.pool_type.lower() == "cls":
            embeddings = ClsPooled
        elif self.pool_type.lower() == "mean":
            embeddings = self.mean_pooling(hiddenState, attention_mask)
        return embeddings

    def max_pooling(self, hidden_state, attention_mask):
        #CLS: First element of model_output contains all token embeddings
        token_embeddings = hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        pooled_embeddings = torch.max(token_embeddings, 1)[0]
        return pooled_embeddings
    
    def mean_pooling (self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) 
        return pooled_embeddings
    


model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
bert_model = RobertaModel.from_pretrained(model_name)



model = CBsmiles(bert_model,'cls')
if_cuda = False
if if_cuda:
    model = model.cuda()


print("Finished Initializing model")
model.eval()

## SMILES STRINGS ###############################################################################
# loading SMILES data
filename = '/mnt/data/mikaela/compound_smiles.csv'
df = pd.read_csv(filename)
df_smiles = df.dropna()

# Retrieved Canonicalized SMILES strings using PUG REST API. No need to standardize SMILES strings

#while True:
for SM in df_smiles['SMILES']:
    #SM = input("SMILES string: ")
    inp_SM = tokenizer.encode(SM)#[i+30700 for i in tokenizer.encode(SM)]
    inp_SM = inp_SM[:min(128, len(inp_SM))]
    inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
    att_SM = torch.ones(inp_SM.shape).long()

# should embeddings be saved to a .csv?
    with torch.no_grad():
         embeddings = model(inp_SM,att_SM)
         print(embeddings)
         print('\n')


