##############################################
# created on: 1/16/2024
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
import pandas as pd
import os
import random
import torch
#from data_process import SelectGenes

class SelectGenes:
      def __init__(self,file_path):
          self.file_path = file_path
          self.knownGenes = []
          self.get_known_genes()
      
      def get_known_genes(self):
          for i in range (14):
              file_path = os.path.join(self.file_path,f'GeneLLM_all_cluster{i}.txt')
              with open(file_path, 'r') as file:
                   for line in file:
                       self.knownGenes.append(line.strip()) 
      
      def filter_for_known_genes(self,input_ids,column_label):
          return input_ids[input_ids[column_label].isin(self.knownGenes)]

      def filter_for_labels(self, input_ids,labels,column_label):
          print(column_label)
          print(labels)
          return input_ids[input_ids[column_label].isin(labels)]
