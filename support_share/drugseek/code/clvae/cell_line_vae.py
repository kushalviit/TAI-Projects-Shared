################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR
import argparse
import sys
import os





class CellLineVAE(nn.Module):
    def __init__(self, dims,dor=0.4):
        super(CellLineVAE, self).__init__()
        encode_list = []
        decode_list = []
        for i in range(len(dims)-1):
            encode_list.append(nn.Linear(dims[i], dims[i+1]))
            encode_list.append(nn.BatchNorm1d(dims[i+1]))
            encode_list.append(nn.ReLU())
            encode_list.append(nn.Dropout(dor))

        for i in range(len(dims)-1, 0, -1):
            decode_list.append(nn.Linear(dims[i], dims[i-1]))
            decode_list.append(nn.BatchNorm1d(dims[i-1]))
            decode_list.append(nn.ReLU())
            #if i != 1:
            decode_list.append(nn.Dropout(dor))
        
        self.encode = nn.Sequential(*encode_list)
        self.decode = nn.Sequential(*decode_list)
        self.fc_mu = nn.Linear(dims[-1], dims[-1])
        self.fc_logvar = nn.Linear(dims[-1], dims[-1])

    def encoder(self, x):
        h4 = self.encode(x)
        mu = self.fc_mu(h4)
        logvar = self.fc_logvar(h4)
        return mu, logvar, h4
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decoder(self, z):
        return self.decode(z)
        
    def forward(self, x):
        mu, logvar, hidden = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, hidden
    
def loss_function( recon_x, x, mu, logvar):
    beta = 1.8
    alpha = 1.2
    MSE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta*MSE + alpha*KLD , MSE, KLD


filename = '../data/processed_OmicsExpression.csv'
df = pd.read_csv(filename)
df = df.dropna()

device = torch.device('cuda' )
cell_line_label= df.columns.to_list()[0]
gene_names = df.columns.to_list()[1:]


norm_df = (df[gene_names]-df[gene_names].min())/(df[gene_names].max()-df[gene_names].min()+1e-4)
#norm_df = (df[gene_names]-df[gene_names].mean())/(df[gene_names].std()+1e-4)
norm_df.insert(0, cell_line_label,df[cell_line_label].values.tolist())

train_df_cl ,temp_df_cl = train_test_split(norm_df, test_size=0.2, random_state=42)
val_df_cl, test_df_cl = train_test_split(temp_df_cl, test_size=0.6, random_state=42)


train_df = train_df_cl[gene_names]
val_df = val_df_cl[gene_names]
test_df = test_df_cl[gene_names]

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)



epochs =  90
batch_size = 32
input_dim = len(gene_names)
latent_dim = 100
dims = [input_dim,10000, 5000, 1000,500,latent_dim]

train_np = train_df.to_numpy()
val_np = val_df.to_numpy()
test_np = test_df.to_numpy()

trainLoader = DataLoader(TensorDataset(torch.tensor(train_np,dtype = torch.float32).to(device)), batch_size=batch_size, shuffle=True)
valLoader = DataLoader(TensorDataset(torch.tensor(val_np,dtype = torch.float32).to(device)), batch_size=batch_size, shuffle=False)
testLoader = DataLoader(TensorDataset(torch.tensor(test_np,dtype = torch.float32).to(device)), batch_size=batch_size, shuffle=False)
drop_rate = 0.2
model = CellLineVAE(dims)
model.to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=35, gamma=0.1)

train_loss = []
val_loss = []
train_corr = []
val_corr = []

for epoch in range(epochs):
    model.train()
    train_loss_epoch = 0
    corr_epoch = 0
    print(f'Epoch: {epoch}')
    for j, batch in enumerate(trainLoader):
        optimizer.zero_grad()
        x = batch[0]
        x.to(device)
        recon_x, mu, logvar, hidden = model(x)
        avg_mu = mu.mean()
        avg_logvar = logvar.mean()
        total_loss, mse,kld = loss_function(recon_x, x, mu, logvar)
        print(f'iteration:{j},Average Mean :{avg_mu}, Average Log_var{avg_logvar}')
        print('MSE: {:.6f} KLD: {:.6f}'.format(mse, kld))
        total_loss.backward()
        optimizer.step()
        train_loss_epoch += total_loss.item()
        corr_epoch += pearsonr(x.to('cpu').detach().numpy().flatten(), recon_x.to('cpu').detach().numpy().flatten())[0]
    scheduler.step()

    train_loss.append(train_loss_epoch/len(trainLoader))
    train_corr.append(corr_epoch/len(trainLoader))
    print('  Train Loss: {:.6f}  ,train_corr: {:.4f}'.format( train_loss_epoch/len(trainLoader), corr_epoch/len(trainLoader)))
    
    model.eval()
    val_loss_epoch = 0  
    val_corr_epoch = 0
    with torch.no_grad():
        for j, batch in enumerate(valLoader):
            x = batch[0]
            x.to(device)
            recon_x, mu, logvar, hidden = model(x)
            total_loss, mse, kld = loss_function(recon_x, x, mu, logvar)
            val_loss_epoch += total_loss.item()
            val_corr_epoch += pearsonr(x.to('cpu').detach().numpy().flatten(), recon_x.to('cpu').detach().numpy().flatten())[0]
        val_corr.append(val_corr_epoch/len(valLoader))
        val_loss.append(val_loss_epoch/len(valLoader))
        print('Val Loss: {:.6f} Val Corr:{:.4f}'.format(val_loss_epoch/len(valLoader), val_corr_epoch/len(valLoader)))
        print('MSE: {:.6f} KLD: {:.6f}'.format(mse, kld))
        print('')

plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
#plt.show()
plt.savefig('../data/loss.png')
plt.figure()
plt.plot(train_corr, label='Train Corr')
plt.plot(val_corr, label='Val Corr')
plt.legend()
#plt.show()
plt.savefig('../data/corr.png')

model.eval()
total_corr = 0
total_loss = 0
for j, batch in enumerate(testLoader):
    x = batch[0]
    x.to(device)
    recon_x, mu, logvar, hidden = model(x)
    total_loss, mse, kld = loss_function(recon_x, x, mu, logvar)
    test_corr = pearsonr(x.to('cpu').detach().numpy().flatten(), recon_x.to('cpu').detach().numpy().flatten())[0]
    total_corr += test_corr
    total_loss += total_loss
    print('MSE: {:.6f} KLD: {:.6f}'.format(mse, kld))

print(f'Test Corr:{total_corr/len(testLoader)}, Total Loss:{total_loss.item()/len(testLoader)}')


torch.save(model.state_dict(), '../data/cell_line_vae.pth')