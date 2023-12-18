##############################################
# created on: 1/16/2024
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
import pandas as pd
import os
#from torch.utils.data import Dataset
import random
import torch
from data_process import SelectGenes

class MajorityClassifier:
      def __init__(self,file,known_gene_file_path):
         self.file_name = file
         self.knownGeneProcess = SelectGenes(known_gene_file_path)
         self.input_pd = pd.read_csv(file)
         self.output_pd = self.input_pd
         self.classes = []
         self.class_label = "None"
         self.gene_label = "None"
      
      def print_head(self):
          print(self.input_pd.head(n=5))

      def print_column_labels(self):
          print(self.input_pd.columns)
      
      def filter_data(self,filter_labels,labels):
          self.gene_label = input("Input the Column label for genes:")
          self.class_label = input("Input class label for genes:")
          self.output_pd = self.knownGeneProcess.filter_for_known_genes(self.input_pd,self.gene_label)
          if filter_labels:
             self.output_pd = self.knownGeneProcess.filter_for_labels(self.output_pd,labels,self.class_label)
      
      def get_majority_accuracy(self):
          unique_classes = self.output_pd[self.class_label].unique()
          length_of_data = len(self.output_pd[self.gene_label])
          num_occurances = self.output_pd[self.class_label].value_counts()
          print(unique_classes)
          max_class = num_occurances.index[0]
          num_max_class = num_occurances [max_class]
          if num_max_class == length_of_data/len(unique_classes):
             print('Equal classes') 
          #print(max_class)
          return length_of_data,max_class,num_max_class,(num_max_class*1.0/length_of_data)*100
