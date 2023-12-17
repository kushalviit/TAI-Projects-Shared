##############################################
# created on: 11/01/2023
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
import argparse
from transformers import *
import torch
from dataexplorer import DataExplorer
from model import IMDBClassifier
from utils import create_data_loader,create_training_tools,train
from keybert import KeyBERT
import time

doc="""The movie was really great.I couldn't take my eyes of each sceen. 
       Cinematography was awesome!
    """

def main(args):
    #prepare dataset
    data_xplorer = DataExplorer(args.data_path, args.test_split, args.seed)
    
    if args.print_head == 'yes':
        data_xplorer.print_head()

    tokenizer = BertTokenizer.from_pretrained(args.pre_trained_model_name)

    #prepare dataloader
    train_data_loader = create_data_loader(data_xplorer.get_train(), tokenizer, args.max_len, args.batch_size)
    val_data_loader = create_data_loader(data_xplorer.get_val(), tokenizer, args.max_len, args.batch_size)
    #test_data_loader = create_data_loader(data_xplorer.get_test(), tokenizer, args.max_len, args.batch_size)
    
    #create model instance
    device = torch.device(args.device)
    model = IMDBClassifier(data_xplorer.get_nclasses(),args.pre_trained_model_name)
    #model = model.to(device)

    if args.mode=='train':
        model = model.to(device)
        #create optimizer scheduler and loss function
        optimizer, scheduler, loss_fn = create_training_tools(args,model,train_data_loader,device)
        
        start = time.time()
        #train the model
        train(args=args,model=model,train_data_loader=train_data_loader,loss_fn=loss_fn,
                optimizer=optimizer,device = device,scheduler=scheduler,
                train_len=data_xplorer.get_train_len(),val_data_loader=val_data_loader,
                val_len=data_xplorer.get_val_len())    
        end = time.time()
        print(f'time need to run on CPU {(end-start)/60}')

    elif args.mode == 'keywordxtract':
        model_path = args.model_path
        model.load_state_dict(torch.load(model_path))
        kw_model = KeyBERT(model.return_bert())
        keywords = kw_model.extract_keywords(doc,highlight=True)
        print(keywords)
    else:
        print("Unknown mode: Use either 'train' or 'keywordxtract'.mode is case sensitive!")






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine Tune bert and visualize keywords.')
    parser.add_argument("--data-path",help="Supply data path to visualize and train", type=str,default='../../data/IMDB_Dataset.csv')
    parser.add_argument("--batch-size",help="Batch size for training", type= int,default=32)
    parser.add_argument("--print-head",help="Print few samples of data", type=str,default='no')
    parser.add_argument("--seed",help="Seed value for randomizing dataset", type= int,default=42)
    parser.add_argument("--test-split",help="Percentage split for training and validation", type= float, default=0.1)
    parser.add_argument("--max-len",help="Max Length of sentance", type= int,default=200)
    parser.add_argument("--epochs",help="Number of epochs for fine tuning", type=int, default=5)
    parser.add_argument("--init-lr",help="Initial learning rate", type=float, default=3e-5)
    parser.add_argument("--optimizer-type",help="Optimizer algorithm and type", type=str, default='adamw')
    parser.add_argument("--device",help="Training on Device cuda or cpu",type=str, default='cpu')
    parser.add_argument("--pre-trained-model-name",help="Name of the pretrained bert model", type=str, default='bert-base-cased')
    parser.add_argument("--mode",help="Mode of opeartion 'train' or 'keywordxtract'.Case Sensitive!", type=str, default='keywordxtract')
    parser.add_argument("--model-path",help="LLM Model Path for extracting keyword", type=str, default='best_model_state.bin')
    args = parser.parse_args()
    main(args)
    #example command to run this code
    #python bert_sentiment_analysis.py --data-path='/taiprojects/support_share/gene_llm_support/visualization/keyword/data/IMDB_Dataset.csv' 
    # --batch-size=8 --seed=42 --test-split=0.1 --max-len=200 --epochs=4 --init-lr=2e-5 --optimizer-type='adamw' --device='cpu' --pre-trained-model-name='bert-base-cased' --print-head='yes'
    # --mode='keywordxtract' --model-path='best_model_state.bin'
