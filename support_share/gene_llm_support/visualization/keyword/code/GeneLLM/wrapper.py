import os
import shutil
import json
from utils import save_finetuned_embeddings
from data_processor import loading_data, process_data
from modeltrainer import trainer
from models import Logistic_Regression


def analyze(input_data_path, task_type, task_name):

    #########################################----------Prepare Data----------------######################################################################


    gene_loaded_data, n_labels = loading_data(input_data_path, task_type)
    print(f'Number of {task_name} labels: {n_labels}')
    print(gene_loaded_data)

    #########################################----------Create necessary directories----------------######################################################################
    if os.path.exists(f"../../data/{task_name}"):
        shutil.rmtree(f"../../data/{task_name}")
    if os.path.exists(f"../../data/{task_name}/enrichment_analysis"):
        shutil.rmtree(f"../../data/{task_name}/enrichment_analysis")

    os.makedirs(f"../../data/{task_name}", exist_ok=True)
    os.makedirs(f"../../data/{task_name}/enrichment_analysis", exist_ok=True)

    ##############################################----------------Split to test & val----------------######################################################################################

    max_length = 512
    batch_size = 40

    train_loader, val_loader, test_loader = process_data(gene_loaded_data, max_length, batch_size, gene2vec_flag = False, model_name = "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
    val_genes = val_loader.dataset.tensors[-1]
    test_genes = test_loader.dataset.tensors[-1]

    data_dict = {
        "val": val_genes.tolist(),
        "test": test_genes.tolist()
    }

    json_file_path = f"../../data/{task_name}/val_test_split_{task_name}.json"

    with open(json_file_path, 'w') as json_file:
        json.dump(data_dict, json_file)

    print(f'Data saved to {json_file_path}')

    ####################################################------------Run Bert large for finetuning without gene2vec--------------################################################################################




    with open(f"../../data/{task_name}/val_test_split_{task_name}.json", 'r') as json_file:
        dict_split = json.load(json_file)

    val_genes = dict_split["val"]
    test_genes = dict_split["test"]
    final_preds = {ind:{0:0,1:0,2:0} for ind in test_genes}

    names = ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"]



    epochs = 2
    lr = 1e-6
    max_length = 512
    batch_size = 20
    pool ="cls"
    drop_rate = 0.1
    gene2vec_hidden = 200
    device = "cuda"
    class_map = None
    n_labels = n_labels


    for model_name in names:
        for gene2vec_flag in [False]:
            
            print(f"model :{model_name}, gene2vec: {gene2vec_flag}")
            
            if gene2vec_flag:
                task="sub_gene2vec"
            else:
                task="class"
            
            train_loader, val_loader, test_loader = process_data(gene_loaded_data, max_length, batch_size,
                                                                    val_genes, test_genes,
                                                                    gene2vec_flag = gene2vec_flag,
                                                                    model_name = model_name)

            history, labels_test, best_pred = trainer(
                                                        epochs, gene_loaded_data, train_loader, val_loader, test_loader,
                                                        lr=lr, pool=pool, max_length=max_length, drop_rate=drop_rate,
                                                        gene2vec_flag=False, gene2vec_hidden=gene2vec_hidden,
                                                        device=device, task_type=task_type,n_labels = n_labels, model_name=model_name,
                                                        task_name=task_name
                                                        )
  ####################################################------------Save finetuned embedddings--------------################################################################################
            save_finetuned_embeddings(gene_loaded_data, pool = pool, max_length= max_length, batch_size = batch_size, drop_rate =drop_rate,
                                      gene2vec_flag = False, gene2vec_hidden=gene2vec_hidden, device=device, task_type=task_type,
                                      n_labels = n_labels, model_name=model_name, task_name=task_name)
    ####################################################------------Logistic regression on the fine tuned data if it is classification--------------################################################################################

    if task_type == 'classification':

        finetuned_emb_path = f'../../data/{task_name}/fine_tuned_embeddings_{task_name}.csv'
        Logistic_Regression(task_name, finetuned_emb_path, input_data_path)


