##############################################
# created on: 11/16/2023
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
import pandas as pd
import os
from torch.utils.data import Dataset
import random
import torch


def get_knowngenes(known_genes_file_path):
    knownGenes = []
    for i in range (14):
        file_path = os.path.join(known_genes_file_path,f'GeneLLM_all_cluster{i}.txt')
        with open(file_path, 'r') as file:
            for line in file:
                knownGenes.append(line.strip()) 
    return knownGenes

class InteactionsProcessor:
    def __init__(self,file_name):
        self.ppi_table = pd.read_csv(file_name)
        self.ppi_table.at[7,'combined_score']=594

    def remove_self_interactions(self):
        self.ppi_table = self.ppi_table[self.ppi_table['protein1']!=self.ppi_table['protein2']]
    
    def remove_unprocessed_genes(self,embeddings_filename):
        processed_genes = pd.read_csv(embeddings_filename)
        processed_genes = processed_genes.iloc[:,0].values.tolist()
        self.ppi_table = self.ppi_table[self.ppi_table['protein1'].isin(processed_genes)]
        self.ppi_table = self.ppi_table[self.ppi_table['protein2'].isin(processed_genes)]

    def mine_ppi(self,embeddings_filename,refined_ppi_file_name):
        print(f'Length of Original ppi {len(self.ppi_table)}')
        self.ppi_table = self.ppi_table.dropna()
        print(f'Length of ppi after droping NA {len(self.ppi_table)}')
        self.remove_self_interactions()
        print(f'Length of ppi after removing self interactions {len(self.ppi_table)}')
        self.remove_unprocessed_genes(embeddings_filename)
        print(f'Lenght of ppi after removing unprocessed genes  {len(self.ppi_table)}')
        self.save_refined_inteactions(refined_ppi_file_name=refined_ppi_file_name)

    
    def save_refined_inteactions(self,refined_ppi_file_name):
        self.ppi_table.to_csv(refined_ppi_file_name)

    
    
class train_test_val_split:
    def __init__(self,file_name,train_percent):
        self.ppi_table = pd.read_csv(file_name)
        self.train_percent = train_percent
    
    def get_interact_genes(self):
        all_unique_genes = self.ppi_table['protein1'].values.tolist()
        all_unique_genes = all_unique_genes + self.ppi_table['protein2'].values.tolist()
        all_unique_genes = list(set(all_unique_genes))
        return all_unique_genes
    
    def get_details(self):
        all_unique_genes = self.get_interact_genes()
        random.shuffle(all_unique_genes)
        train_length = int(len(all_unique_genes)*self.train_percent/100)
        train_length =round(train_length,-2)
        val_length = int(train_length *0.1)
        train_length = train_length - val_length
        return  all_unique_genes[0:train_length],all_unique_genes[train_length:train_length+val_length],all_unique_genes[train_length+val_length:]


    

class siameseDataset(Dataset):
    
    def __init__(self,file_name,summary_file_name,usable_genes,non_usable_genes):
        self.ppi_table = pd.read_csv(file_name)
        self.summary = pd.read_csv(summary_file_name,header=None)
        self.usable_genes = usable_genes
        self.non_usable_genes = non_usable_genes
        self.total_gene_set = set(self.usable_genes).union(set(self.non_usable_genes))
        self.alternator = 0
    
    def get_interaction_details(self,gene):
        interactions_dict={}
        protein2_interaction=self.ppi_table[self.ppi_table['protein1']==gene]
        protein2_interaction=set(protein2_interaction['protein2'].values.tolist())
        protein1_interaction=self.ppi_table[self.ppi_table['protein2']==gene]
        protein1_interaction=set(protein1_interaction['protein1'].values.tolist())
        interacting_proteins = protein1_interaction.union(protein2_interaction)
        uniteracting_proteins =  self.total_gene_set- interacting_proteins 
        uniteracting_proteins= uniteracting_proteins-set(self.non_usable_genes)
        protein1_interaction = protein1_interaction - set(self.non_usable_genes)
        protein2_interaction = protein2_interaction - set(self.non_usable_genes)
        interactions_dict["interacting"] = list(interacting_proteins)        
        interactions_dict["uninteracting"] = list(uniteracting_proteins)
        return interactions_dict
    
    def gene_swap(self, g0,g1):
        return g1,g0

    def __getitem__(self,index):
        gene0 = self.usable_genes[index]
        interact_dict = self.get_interaction_details(gene0)
        swap = random.randint(0,1)

        if self.alternator ==0:
           class_type = 0
           self.alternator =1
           idx = random.randint(0,len(interact_dict["uninteracting"])-1)
           gene1 = interact_dict["uninteracting"][idx]
        elif self.alternator == 1:
            self.alternator =0
            class_type = 1
            idx = random.randint(0,len(interact_dict["interacting"])-1)
            gene1 = interact_dict["interacting"][idx]
 
            
        if swap == 1:
            gene0, gene1= self.gene_swap(gene0,gene1)

        genes0=self.summary[self.summary.iloc[:,0]==gene0].iloc[0,1:].values.tolist()
        genes1=self.summary[self.summary.iloc[:,0]==gene1].iloc[0,1:].values.tolist()
        genes0 = torch.Tensor(genes0)
        genes1 = torch.Tensor(genes1)
        class_type = torch.Tensor([class_type])

        return genes0, genes1, class_type 
    
    def __len__(self):
        return len(self.usable_genes)
    


class contrastiveTrainingDataset(Dataset):
    
    
    def __init__(self,file_name,summary_file_name,usable_genes,non_usable_genes):
        self.ppi_table = pd.read_csv(file_name)
        self.summary = pd.read_csv(summary_file_name,header=None)
        self.usable_genes = usable_genes
        self.non_usable_genes = non_usable_genes
        self.total_gene_set = set(self.usable_genes).union(set(self.non_usable_genes))
        self.alternator = 0
    
    def get_interaction_details(self,gene):
        interactions_dict={}
        protein2_interaction=self.ppi_table[self.ppi_table['protein1']==gene]
        protein2_interaction=set(protein2_interaction['protein2'].values.tolist())
        protein1_interaction=self.ppi_table[self.ppi_table['protein2']==gene]
        protein1_interaction=set(protein1_interaction['protein1'].values.tolist())
        interacting_proteins = protein1_interaction.union(protein2_interaction)
        uniteracting_proteins =  self.total_gene_set- interacting_proteins 
        uniteracting_proteins= uniteracting_proteins-set(self.non_usable_genes)
        protein1_interaction = protein1_interaction - set(self.non_usable_genes)
        protein2_interaction = protein2_interaction - set(self.non_usable_genes)
        interactions_dict["interacting"] = list(interacting_proteins)        
        interactions_dict["uninteracting"] = list(uniteracting_proteins)
        return interactions_dict
    
    def gene_swap(self, g0,g1):
        return g1,g0

    def __getitem__(self,index):
        gene0 = self.usable_genes[index]
        interact_dict = self.get_interaction_details(gene0)
        swap = random.randint(0,1)


        nidx = random.randint(0,len(interact_dict["uninteracting"])-1)
        gene1p = interact_dict["uninteracting"][nidx]
        

        class_type = 1
        pidx = random.randint(0,len(interact_dict["interacting"])-1)
        gene1n = interact_dict["interacting"][pidx]
 
            
        if swap == 1:
            gene0, gene1= self.gene_swap(gene0,gene1)

        genes0=self.summary[self.summary.iloc[:,0]==gene0].iloc[0,1:].values.tolist()
        genes1p=self.summary[self.summary.iloc[:,0]==gene1p].iloc[0,1:].values.tolist()
        genes1n=self.summary[self.summary.iloc[:,0]==gene1n].iloc[0,1:].values.tolist()
        genes0 = torch.Tensor(genes0)
        genes1p = torch.Tensor(genes1p)
        genes1n = torch.Tensor(genes1n)

        return genes0, genes1p, gene1n
    
    def __len__(self):
        return len(self.usable_genes)