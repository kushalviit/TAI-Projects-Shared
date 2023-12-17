##############################################
# created on: 11/16/2023
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
import sys
sys.path.append("../../code/GeneLLM/")
from models import FineTunedBERT
import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from data_processor import process_sent
import numpy as np
from data_process_load import get_knowngenes



class EmbeddingXtractor:
    def __init__(self, args,gene2vec_flag):
        self.model =  FineTunedBERT(pool= args.pool, task_type = 'classification', n_labels = 3,
                          drop_rate = args.drop_rate, model_name = args.bert_model_name,
                          gene2vec_flag= gene2vec_flag,
                          gene2vec_hidden = args.gene2vec_length)
        if args.bert_embedding_type =="fine_tuned":
            state_dict = torch.load(args.fine_tuned_state_dict)
            self.model.load_state_dict(state_dict)
        self.device = args.device
        self.model = self.model.to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
        self.max_length = args.max_length
        self.KnownGenes = get_knowngenes(args.known_genes_file_path)
        self.genes = self.get_summary(args.summary_file_name)
        self.batch_size = args.batch_size
        self.embeddings_filename = args.embedding_file_name



    def get_summary(self,summary_file_name):
        genes = pd.read_csv(summary_file_name)
        genes["Summary"] = genes["Summary"].apply(lambda sent: process_sent(sent))
        genes = genes.drop_duplicates(subset='Summary')
        genes = genes[genes['Gene name'].isin(self.KnownGenes)]
        return genes


    def save_embeddings(self): 
        encoded_summaries = self.tokenizer.batch_encode_plus(self.genes["Summary"].tolist(), max_length=self.max_length, padding="max_length",
                                                   truncation=True,
                                                   return_tensors="pt")

        # DataLoader for all genes
        all_dataset = TensorDataset(encoded_summaries["input_ids"], encoded_summaries["attention_mask"])
        all_data_loader = DataLoader(all_dataset, batch_size=self.batch_size, shuffle=False)

        # Store gene names separately
        all_gene_names = self.genes["Gene name"].tolist()

        # Get embeddings for all genes
        all_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, masks) in enumerate(all_data_loader):
                embeddings, _, _ =self.model(inputs.to(self.device), masks.to(self.device))
                all_embeddings.append(embeddings.cpu().numpy())

            # Flatten embeddings list
            all_embeddings = np.vstack(all_embeddings)

        embeddings_df = pd.DataFrame(all_embeddings)
        embeddings_df['Gene name'] = all_gene_names  

        embeddings_df = pd.concat([embeddings_df.iloc[:, -1], embeddings_df.iloc[:, :-1]], axis=1)
        embeddings_df.columns = [''] * len(embeddings_df.columns)
        embeddings_df.to_csv(self.embeddings_filename,header=False,index=False)
        print(f'Embeddings saved to {self.embeddings_filename}')


