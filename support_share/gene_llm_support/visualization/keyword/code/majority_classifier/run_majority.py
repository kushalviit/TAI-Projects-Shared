##############################################
# created on: 1/16/2024
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################

import argparse
import sys
from majority_classifier import MajorityClassifier


def main(args):
    print(f"Processing data for {args.task_type}:")
    majority_classifier = MajorityClassifier(args.file_name,args.file_path)
    majority_classifier.print_head()
    majority_classifier.print_column_labels()
    
    labels=[]
    filter_label=False
    if args.filter_labels == "yes":
       continue_inp ='y'
       while(continue_inp == 'y'):
           tmp = input("Enter the label to be retained:")
           labels.append(tmp)
           continue_inp = input("Do you want to input labels(y/n):")
       filter_label = True
    majority_classifier.filter_data(filter_label,labels)
    #majority_classifier.get_majority_accuracy()
    datalen,max_label,num_max,accuracy = majority_classifier.get_majority_accuracy()
    print(f"Data Length for {args.task_type}: {datalen} ")
    #print(f"Class labels in {args.task_type}: {unique_labels}")
    print(f"Max class label for {args.task_type}: {max_label}")
    print(f"Num. of data points for label {max_label}: {num_max}")
    print(f"Total Accuracy: {accuracy}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Majority Classifier for GeneLLM.')
    parser.add_argument("--file-name", help="Enter file name for the task", type=str, default='../../data/genellm_data/solubility.csv')
    parser.add_argument("--file-path", help="Enter file path for the known genes",type =str,default = '../../data/knownGenes/')
    parser.add_argument("--filter-labels", help="Enter yes if filtering is needed for labels",type = str,default = "no")
    parser.add_argument("--task-type", help="Enter task type", type=str, default='Solubility')
    args = parser.parse_args()
    main(args)






