from models import FineTunedBERT , MultiLabelFocalLoss
from utils import train, validation, test, get_metrics, plot_latent
from transformers import AdamW
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch
import pandas as pd
import copy


def trainer(epochs, genes, train_loader, val_loader, test_loader,
                lr = 1e-5, pool = "cls", max_length= 100, batch_size =100, drop_rate =0.1,
                gene2vec_flag = False, gene2vec_hidden = 200, device = "cuda",
                task_type = "classification", n_labels = 3 , model_name= "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract", task_name = 'Subcellular_location'):
    
    
    """
        gene2vec_flag: if True then, the embeddings of gene2vec will be concat to GeneLLM embeddings.

        model_name: "xlnet-base-cased",
                    "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
                    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "dmis-lab/biobert-base-cased-v1.1",
                    "bert-base-cased",
                    "bert-base-uncased"


        task_type = classification or regression
    
    
    """
    
    

    #subcell : {0:'Cytoplasm', 1:'Nucleus', 2:'Cell membrane'}
    #Sol: {0:'Membrane', 1:'Soluble'}
    #cons: {0:}
    global class_map
    if task_type == "classification":
        history = {
            "Train":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Val":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Test":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]}}
        
        unique_values = genes.iloc[:, 3].unique()
        
        class_map = {i: value for i, value in enumerate(unique_values)}
        print ("\n#############################")
        print (f"Currently running {task_name}.")
        print ("#############################\n")


    elif task_type == "multilabel":
        history = {
            "Train":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Val":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Test":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]}}
        
            
        print ("\n#############################")
        print ("Currently running {task_name}.")
        print ("#############################\n")          



    elif task_type == "regression":
        history = {
            "Train":{"Correlation":[]}, "Val":{"Correlation":[]}, "Test":{"Correlation":[]}}
        
        if n_labels == 1:
            
            class_map = None
            print ("\n###############################")
            print (f"Currently running {task_name}.")
            print ("###############################\n")
        
    

    if task_type == "regression":
        loss_fn = nn.MSELoss()
        
    elif task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
        
    elif task_type == "multilabel":
        loss_fn = MultiLabelFocalLoss()
        # loss_fn = nn.BCELoss()
    else:
        raise ValueError(f"task type error: {task_type}")




    model = FineTunedBERT(pool= pool, task_type = task_type, n_labels = n_labels,
                          drop_rate = drop_rate, model_name = model_name,
                          gene2vec_flag= gene2vec_flag,
                          gene2vec_hidden = gene2vec_hidden)
    
    model.assign_bert_to_device()

    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
#     train_loader, val_loader, test_loader = process_data_sub(genes, max_length, batch_size,
#                                                              gene2vec_flag = gene2vec_flag,
#                                                              model_name = model_name)
    
    best_pred = None
    optimal_acc = -1
    global best_model_state
    best_model_state = None
    best_val_metric = 0.0 if task_type in ["classification", "multilabel"] else float('-inf')




    
    
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"Epoch {epoch+1} of {epochs}")
        print("-------------------------------")
        
        print("Training ...")
        
        model, train_loss, labels_train, pred_train, latents = train(train_loader, model, loss_fn, optimizer,
                                                                     task_type = task_type,
                                                                     gene2vec_flag = gene2vec_flag,
                                                                     device = device)
        print(latents.size())
        plot_latent(latents, labels_train,  epoch, class_map, task_name, validation_type="train")
        
        
        
        print("Validation ...")
        model, val_loss, labels_val, pred_val  = validation (val_loader, model, loss_fn,
                                                             task_type = task_type,
                                                             gene2vec_flag = gene2vec_flag,
                                                             device = device)
        
        print("Testing ...")
        model, test_loss, labels_test, pred_test, _ = test (test_loader, model, loss_fn,
                                                            task_type = task_type,
                                                            gene2vec_flag = gene2vec_flag,
                                                           device = device)
        

        metrics_train  = get_metrics(labels_train , pred_train, history,
                                      val_type = "Train", task_type = task_type)
        
        metrics_val = get_metrics(labels_val , pred_val, history,
                                  val_type = "Val",task_type = task_type)
    
        
        

        metrics_test = get_metrics(labels_test , pred_test, history, val_type = "Test",
                                   task_type = task_type)

        if task_type == "classification" or task_type == "multilabel":
            acc_train, f1_train, prec_train, rec_train = metrics_train
            acc_val, f1_val, prec_val, rec_val = metrics_val
            acc_test, f1_test, prec_test, rec_test = metrics_test

            

            print(f'\tET: {round(time.time() - start_time,2)} Seconds')
            print(f'Train Loss: {round(train_loss,4)}, Accuracy: {round(acc_train,4)}, F1: {round(f1_train,4)}, Precision: {round(prec_train,4)}, Recall: {round(rec_train,4)}')
            print(f'Val Loss: {round(val_loss,4)}, Accuracy: {round(acc_val,4)}, F1: {round(f1_val,4)}, Precision: {round(prec_val,4)}, Recall: {round(rec_val,4)}')
            print(f'Test Loss: {round(test_loss,4)}, Accuracy: {round(acc_test,4)}, F1: {round(f1_test,4)}, Precision: {round(prec_test,4)}, Recall: {round(rec_test,4)}')
    
            with open(f'../../data/{task_name}/metrics_{task_name}.csv', 'a') as f:
                if epoch == 0:
                    f.write("epoch,train_loss,train_acc,train_f1,train_prec,train_rec,val_loss,val_acc,val_f1,val_prec,val_rec,test_loss,test_acc,test_f1,test_prec,test_rec\n")
                f.write(f"{epoch+1},{train_loss},{acc_train},{f1_train},{prec_train},{rec_train},{val_loss},{acc_val},{f1_val},{prec_val},{rec_val},{test_loss},{acc_test},{f1_test},{prec_test},{rec_test}\n")
        else:
            
            train_corr = metrics_train
            val_corr = metrics_val
            test_corr = metrics_test
            
            print(f'\tET: {round(time.time() - start_time,2)} Seconds')
            print(f'\tTrain Loss: {round(train_loss,4)}, corrcoef: {round(train_corr,4)}')
            print(f'\tVal Loss: {round(val_loss,4)}, corrcoef: {round(val_corr,4)}')
            print(f'\tTest Loss: {round(test_loss,4)}, corrcoef: {round(test_corr,4)}')
            
            with open(f'../../data/{task_name}/metrics_{task_name}.csv', 'a') as f:
                if epoch == 0:
                    f.write(f"epoch,train_loss,train_corr,val_loss,val_corr,test_loss,test_corr\n")
                f.write(f"{epoch+1},{train_loss},{train_corr},{val_loss},{val_corr},{test_loss},{test_corr}\n")



        current_val_metric = val_corr if task_type == "regression" else acc_val  
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, f'../../data/{task_name}/best_model_{task_name}.pth')
            global best_epoch_num
            best_epoch_num = epoch + 1
            print(f'best_epoch_num: {best_epoch_num}')

        #plot the losses and accuracies till best epoch
        if epoch == len(range(epochs))-1:
            if task_type == "regression":
                df_corr = pd.read_csv(f'../../data/{task_name}/metrics_{task_name}.csv')
                df_filtered = df_corr[df_corr['epoch'] <= best_epoch_num]
                plt.figure(figsize=(12, 5))

                # Plot for Loss
                plt.subplot(1, 2, 1)
                plt.plot(df_filtered['epoch'], df_filtered['train_loss'], label='Train Loss', marker='o')
                plt.plot(df_filtered['epoch'], df_filtered['val_loss'], label='Validation Loss', marker='o')
                plt.plot(df_filtered['epoch'], df_filtered['test_loss'], label='Test Loss', marker='o')
                plt.title('Loss across Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)

                # Plot for Correlation Coefficients
                plt.subplot(1, 2, 2)
                plt.plot(df_filtered['epoch'], df_filtered['train_corr'], label='Train Correlation', marker='o')
                plt.plot(df_filtered['epoch'], df_filtered['val_corr'], label='Validation Correlation', marker='o')
                plt.plot(df_filtered['epoch'], df_filtered['test_corr'], label='Test Correlation', marker='o')
                plt.title('Correlation Coefficients across Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Correlation Coefficient')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'../../data/{task_name}/finetuning_loss_{task_name}.png')

            elif task_type == "classification" or task_type == "multilabel":
                df = pd.read_csv(f'../../data/{task_name}/metrics_{task_name}.csv')
                df_filtered = df[df['epoch'] <= best_epoch_num]
                plt.figure(figsize=(12, 5))

                # Plot for Loss
                plt.subplot(1, 2, 1)
                plt.plot(df_filtered['epoch'], df_filtered['train_loss'], label='Train Loss', marker='o')
                plt.plot(df_filtered['epoch'], df_filtered['val_loss'], label='Validation Loss', marker='o')
                plt.plot(df_filtered['epoch'], df_filtered['test_loss'], label='Test Loss', marker='o')
                plt.title('Loss across Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)

                # Plot for Accuracy
                plt.subplot(1, 2, 2)
                plt.plot(df_filtered['epoch'], df_filtered['train_acc'], label='Train Accuracy', marker='o')
                plt.plot(df_filtered['epoch'], df_filtered['val_acc'], label='Validation Accuracy', marker='o')
                plt.plot(df_filtered['epoch'], df_filtered['test_acc'], label='Test Accuracy', marker='o')
                plt.title('Accuracy across Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'../../data/{task_name}/finetuning_loss_{task_name}.png')
                



            


    return history, labels_test, best_pred