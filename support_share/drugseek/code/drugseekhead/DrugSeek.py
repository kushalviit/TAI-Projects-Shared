from chemberta_drugseek import CBsmiles
from vae_new import VAE 
import torch
import torch.nn as nn
import torch.optim as optim

class DrugSeek(nn.Module):
    def __init__(self, CLvae, CB_smiles):
        super(VAE_encoder, self).__init__()
        dims = [1900, 5000, 3000, 1500, 750] # verify these dimensions are correct
        self.CLvae=CLvae
        self.CB_smiles=CB_smiles
        self.fc1 = nn.Linear(dims[0]+dims[1], dims[2]) # input dimensions AND output dimenstions
        self.fc2 = nn.Linear(dims[2], dims[3])
        self.fc3 = nn.Linear(dims[3], dim[4])
        self.fc4 = nn.Linear(dims[4], 1) # the 1 means just one output
        linear_head = [self.fc1, self.fc2, self.fc3, self.fc4]
        head = []
        for i in range(4):
            head.append(linear_head[i])
            head.append(nn.BatchNorm1d(dims[i+2]))
            if i != 3:
                head.append(nn.ReLU())
        self.head = nn.Sequential(*head) # * is a "pointers", must be iterable, and is necessary in pytorch
    
    def encode(self, CLinput, smiles_input_ids, smiles_attention_mask):
        mu, logvar, CLvae_embeddings = self.CLvae.encode(CLinput)
        smiles_embeddings = self.CB_smiles(input_ids = smiles_input_ids,attention_mask=smiles_attention_mask).values()
        return CLvae_embeddings, smiles_embeddings
    
    def forward(self, CLinput, smiles_input_ids, smiles_attention_mask):
        CLvae_embeddings, smiles_embeddings = self.encode(CLinput, smiles_input_ids, smiles_attention_mask)
        input_head = torch.cat((CLvae_embeddings, smiles_embeddings), 1) # 1 means we are concatenating horizontally
        output = self.head(input_head)
        return output



input_dim = 14000
latent_dim = 1900
CLvae = VAE(input_dim, latent_dim)

CLencoder = VAE_encoder(CLvae=CLvae)

    
