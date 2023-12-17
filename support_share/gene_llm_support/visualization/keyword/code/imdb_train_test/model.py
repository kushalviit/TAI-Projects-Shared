import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *

class IMDBClassifier(nn.Module):

  def __init__(self, n_classes,pre_trained_model_name):
    super(IMDBClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(pre_trained_model_name,output_attentions=True)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    lhs, pooled_output,attentions = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,return_dict=False
    )
    output = self.drop(pooled_output)
    return self.out(output),attentions

  def return_bert(self):
      return self.bert
