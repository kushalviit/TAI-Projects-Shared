##############################################
# created on: 11/16/2023
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
from data_process_load import InteactionsProcessor
from embeddings_xtractor import EmbeddingXtractor
from train import train_wrapper
from test import *
import argparse
import sys
from model import SiameseNetwork

def main(args):
    if args.mode_of_operation == 'embeddings_xtractor':
        if args.use_gene2vec == True:
            gene2vec_flag=True
        else:
            gene2vec_flag=False
            args.gene2vec_length=0

        print('Initializing Embedding Extractor ........')
        embedding_extractor=EmbeddingXtractor(args,gene2vec_flag=gene2vec_flag)
        print('Finished Initializing Embedding Extractor. Saving Embeddings .......')
        embedding_extractor.save_embeddings()
        print('Finished saving embeddings')
    elif args.mode_of_operation == 'process_ppi':
        print('Initializing Protein Protein Interaction files ......')
        ppi_preprocessor = InteactionsProcessor(file_name=args.ppi_original_file_name)
        print('Finished Initializing. Beginning Mining PPI ....')
        ppi_preprocessor.mine_ppi(embeddings_filename=args.embedding_file_name,refined_ppi_file_name=args.ppi_refined_file_name)
        print('Finished mining PPI.')
    elif args.mode_of_operation == 'train':
        dims =[1024,512,256,128]
        model=SiameseNetwork(dims=dims,drop_prob=args.drop_rate)
        train_wrapper(model=model,refined_ppi_file_name=args.ppi_refined_file_name,
                      summary_file_name=args.embedding_file_name,
                      num_workers=args.num_workers,batch_size=args.batch_size,
                      num_epochs=args.num_epochs,device=args.device,
                      learning_rate=args.learning_rate,train_percent=args.train_percent,
                      model_save_path=args.model_save_checkpoint)
    elif args.mode_of_operation == 'test':
        print('test')
        exit()
    else:
        sys.exit("Unknown mode of Operations!")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Siamese or Contrastive learning for PPI.')
    parser.add_argument("--mode-of-operation", help="Provide mode of operation [train, inference, embeddings_xtractor, process_ppi]", type=str, default='embeddings_xtractor')
    parser.add_argument("--pool",help="Pooling taken from token",type=str,default='cls')
    parser.add_argument("--device",help="Device on which models and data need to be operated",type=str,default='cuda')
    parser.add_argument("--max-length",help="Maximum length of sentence in bert",type=int, default=100)
    parser.add_argument("--batch-size",help="Batch size for data loader",type=int, default=10)
    parser.add_argument("--num-epochs",help="Number of epochs for training ",type=int, default=50 )
    parser.add_argument("--num-workers",help="Number of threads for dataloader ",type=int, default=15)
    parser.add_argument("--train-percent",help="split data to train ",type=int, default=70)
    parser.add_argument("--drop-rate",help="Batch size for data loader",type=float, default=0)
    parser.add_argument("--learning-rate",help="learning rate for optimizer",type=float, default=0.001)
    parser.add_argument("--bert-model-name",help="Bert model name",type=str,default="microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
    parser.add_argument("--bert-embedding-type",help="Bert embedding of fine tuned or pretrained",type=str,default="pretrained")
    parser.add_argument("--fine-tuned-state-dict",help="Bert embedding of fine tuned or pretrained",type=str,default="../../data/state_dict")
    parser.add_argument("--embedding-file-name",help="name of embeddings file with the path",type=str,default="../../data/embeddings_xtract/gene_embeddings_known_genes.csv")
    parser.add_argument("--summary-file-name",help="name of gene summary file with the path",type=str,default="../../data/genes.csv")
    parser.add_argument("--known-genes-file-path",help="name of gene summary file with the path",type=str,default="../../data/knownGenes/")
    parser.add_argument("--use-gene2vec",help="True/False if gene to vec is being used for embeddings",type=str,default="False")
    parser.add_argument("--gene2vec-length",help="Length of Gene to vec",type=int,default=200)
    parser.add_argument("--ppi-original-file-name",help="Original PPI file name with path",type=str,default="../../data/protein_interactions.csv")
    parser.add_argument("--ppi-refined-file-name",help="Refined PPI file name with path",type=str,default="../../data/refined_protein_interactions_2.csv")
    parser.add_argument("--model-save-checkpoint",help="model checkpoints path",type=str,default="../../model_checkpoints.pth")
    args = parser.parse_args()
    main(args)