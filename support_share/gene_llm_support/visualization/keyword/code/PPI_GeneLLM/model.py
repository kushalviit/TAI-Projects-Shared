##############################################
# created on: 11/16/2023
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
import torch.nn as nn
import torch.nn.functional as F
import torch

class SiameseNetwork(nn.Module):
    def __init__(self,dims,drop_prob):
        super(SiameseNetwork, self).__init__()
        #dims =[1024,512,256,128]
        encoder_layers = []
        for i in range(len(dims)-1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            encoder_layers.append(nn.BatchNorm1d(dims[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)


        self.fc1 = nn.Linear(2*dims[-1], dims[-1])
        self.fc2 = nn.Linear(dims[-1], int(dims[-1]/2))
        self.fc3 = nn.Linear(int(dims[-1]/2), 1)


    def forward(self, input1, input2):
        output1 = self.encoder(input1)
        output2 = self.encoder(input2)
        
        output = torch.cat((output1, output2),1)
        output =F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.sigmoid(self.fc3(output))
        return output