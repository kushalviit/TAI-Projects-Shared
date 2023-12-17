import sys
import torch
##change if you change the folder structure
sys.path.append('../../code/GeneLLM')
from models import FineTunedBERT

def bert_backbone_from_genellm(args):
    if args.gene2vec_flag=='true':
        gene2vec_flag=True
    else:
        gene2vec_flag=False
    model = FineTunedBERT(pool= args.pool, task_type = args.task_type,  
                        n_labels = args.n_labels,drop_rate = args.drop_rate, 
                        model_name = args.model_name, 
                        gene2vec_flag= gene2vec_flag, 
                        gene2vec_hidden = args.gene2vec_hidden)
    
    model.assign_bert_to_device()

    if args.train_type == "finetuned":
        model.load_state_dict(torch.load(args.model_path))
    bert = model.get_bert()
    return bert