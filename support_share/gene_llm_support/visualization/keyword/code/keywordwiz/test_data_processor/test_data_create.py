import pandas as pd
from random import randint
import sys


class DataSetHandler:
    def __init__(self,input_data_path,gene_summary_path,random_summary):
        self.random_summary = random_summary
        self.task_data_frame = pd.read_csv(input_data_path)
        self.gene_summary_frame = pd.read_csv(gene_summary_path)
        self.num_of_test_genes = self.task_data_frame.shape[0]
        self.random_gene_name = self.get_random_gene()
        print(f' Summary of Gene Name -{self.random_gene_name}:')
        
    
    def get_random_gene(self):
        if self.random_summary ==1:
            row_index = randint(0,self.num_of_test_genes-1)
        elif self.random_summary == 0:
            row_index = 42 # need to parametrize it
        else:
            sys.exit("Input argument 'random_summary' is not desired value")
        return self.task_data_frame.loc[row_index,'GeneSymbol']
      
    def get_gene_summary(self):
        idx = self.gene_summary_frame[self.gene_summary_frame['Gene name']== self.random_gene_name].index.values[0]
        summary = self.gene_summary_frame.loc[idx,'Summary']
        return summary