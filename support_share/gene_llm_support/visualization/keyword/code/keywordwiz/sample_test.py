##############################################
# created on: 11/16/2023
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################
import argparse
from keybertmod.xtractor import xtract_keyword
from test_data_processor.test_data_create import DataSetHandler
from utils import model_handler

def main(args):
    backbone_bert = model_handler.bert_backbone_from_genellm(args)

    testds=DataSetHandler(args.data_path,args.gene_summary_path,args.random_summary)
    summary=testds.get_gene_summary()

    keywords_score = xtract_keyword(backbone_bert, summary, args)
    print(keywords_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize keywords.')
    parser.add_argument("--data-path",help="Supply data path to visualize and train", type=str,default='../../data/subcellular_location.csv')
    parser.add_argument("--gene-summary-path",help="Supply data path to visualize and train", type=str,default='../../data/genes.csv')
    parser.add_argument("--model-path",help="Model Path", type=str,default='../../data/subcellular_localization/best_model_subcellular_localization.pth')
    parser.add_argument("--random-summary",help="1 if random gene summary needs visualization 0 if fixed summary",type=int,default=0)
    parser.add_argument("--train-type",help="Fine Tuned or not", type=str,default='finetuned')
    parser.add_argument("--task-type",help="Type of GeneLLM task", type=str, default='classification')
    parser.add_argument("--pool",help="Pooling for GeneLLM", type=str, default='cls')
    parser.add_argument("--drop-rate",help="Dropout rate for GeneLLM. Not needed for Keyword extractor", type=float, default=0.1)
    parser.add_argument("--gene2vec-hidden",help="Gene2Vec Hidden layer size", type=int, default=0)
    parser.add_argument("--gene2vec-flag",help="True if Gene2Vec is needed", type=str, default='true')
    parser.add_argument("--n-labels",help="Number of layers ",type=int, default=3)
    parser.add_argument("--model-name",help="Name of the model",type=str,default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--keybert-type",help="Keybert type",type=str,default="basic")
    parser.add_argument("--ngram-max",help="Keyphrase of length max",type=int,default=1)
    parser.add_argument("--ngram-min",help="Keyphrase of length min",type=int,default=1)
    parser.add_argument("--stop-words",help="Keybert stop words",type=str,default=None)
    parser.add_argument("--nr-candidate",help="Number of candidates chosen for combination of Top N Keywords with similar cosine value",type=int, default=20)
    parser.add_argument("--n-top",help="Top N candidates for Maximum Marginal Relevenace-maxrelavence and Maximum Sum Distance-maxdis mode",type=int, default=5)
    parser.add_argument("--diversity",help="Diversity in cosine similarity",type=float, default=0.7)
    args = parser.parse_args()
    main(args)
    #print(f'pool:{pool} , model_name:{model_name}, task_type:{task_type}, n_labels:{n_labels}, drop_rate:{drop_rate}, gene2vec_flag:{gene2vec_flag},gene2vec_hidden :{gene2vec_hidden} ')


