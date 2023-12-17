import torch
from models import FineTunedBERT
from transformers import BertTokenizerFast
from transformers import XLNetTokenizer
from torch.utils.data import DataLoader, TensorDataset
import warnings
import pandas as pd
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def getEmbeddings(text,
                  model = None,
                  model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                  max_length=512,
                  batch_size=1000,
                  pool ="mean"):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    if model:
        #GeneLLM embeddings
        #model has to be trained on a task to get task specific embeddings
        model = model.to(device)
        
        if "xlnet" in model.model_name:
            tokenizer = XLNetTokenizer.from_pretrained(model.model_name)
        else:    
            tokenizer = BertTokenizerFast.from_pretrained(model.model_name)
            
    else:
        #BERT-base embeddings
        model = FineTunedBERT(pool= pool,
                              model_name=model_name,
                              task_type="classification",
                              gene2vec_flag = False,
                              n_labels = 2).to(device)
        if "xlnet" in model_name:
            tokenizer = XLNetTokenizer.from_pretrained(model_name)
        else:    
            tokenizer = BertTokenizerFast.from_pretrained(model_name)


    print("Tokenization ...")
    tokens = tokenizer.batch_encode_plus(text, max_length = max_length,
                                         padding="max_length",truncation=True,
                                         return_tensors="pt")
    
    
    dataset = TensorDataset(tokens["input_ids"] , tokens["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Tokenization Done.")
    
    print("Get Embeddings ...")
    
    embeddings=[]
    model.eval()
    for batch_input_ids, batch_attention_mask in tqdm(dataloader):
        with torch.no_grad():
            pooled_embeddings, _, _ = model(batch_input_ids.to(device) ,
                                            batch_attention_mask.to(device))
            embeddings.append(pooled_embeddings)
    
    
    concat_embeddings = torch.cat(embeddings, dim=0)
    
    print(concat_embeddings.size())
    
    return concat_embeddings


def getSentenceEmbeddings(sentences, max_length=512, batch_size=1000, pool ="mean"):
    
#     pool="mean", model_name= "bert-base-cased",
#                  task_type = None, n_labels = None, drop_rate = None,

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name= "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    model = FineTunedBERT(pool= pool,
                          model_name=model_name,
                          task_type="classification",
                          gene2vec_flag = False,
                          n_labels = 2).to(device)
    
    tokenizer = BertTokenizerFast.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")

    print("Perform tokenization ...")
    tokens = tokenizer.batch_encode_plus(sentences, max_length = max_length,
                                         padding="max_length",truncation=True,
                                         return_tensors="pt")
    
    
    dataset = TensorDataset(tokens["input_ids"] , tokens["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Tokenization Done.")
    
    print("Get embeddings ...")
    
    embeddings=[]
    model.eval()
    for batch_input_ids, batch_attention_mask in tqdm(dataloader):
        with torch.no_grad():
            pooled_embeddings, _, _ = model(batch_input_ids.to(device) ,
                                            batch_attention_mask.to(device))
            embeddings.append(pooled_embeddings)
    
    
    concat_embeddings = torch.cat(embeddings, dim=0)
    
    print(concat_embeddings.size())
    
    return concat_embeddings

def train(loader, model, loss_fn, optimizer, task_type= None,
          gene2vec_flag = False, device = "cuda",
          threshold=0.5 ):
    """
        task : conservation , sublocation, solubility
        task_type : regression or classification
    
    
    """
    
    train_loss = 0
    latents  = []

    total_preds = []
    total_labels = []
    
    # conservation , sublocation, solubility
   

    
    model.train()
    for batch in loader: 

        
        if gene2vec_flag:
            batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
            embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)
            
            
        else:
            batch_inputs, batch_masks , labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
            embeddings, _ , preds = model(batch_inputs, batch_masks)

        
        
        if task_type == "regression":
            
            preds = preds.squeeze().float()
            labels = labels.squeeze().float()
            
            loss = loss_fn(preds, labels) 
            
            
            total_preds.extend(preds.cpu().detach())
            total_labels.extend(labels.cpu().detach())        

        
        elif task_type == "classification":
            
            loss = loss_fn(preds, labels)
            
            total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
            total_labels.extend(labels.type(torch.int).to('cpu').numpy())

        
        
        elif task_type == "multilabel":

            preds = preds.to(torch.float32)
            labels = labels.to(torch.float32)

            loss = loss_fn(preds, labels)
            
            total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
            total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())
            
        
        train_loss += loss.item()

        
        #Aggregation
        embeddings = torch.tensor(embeddings.cpu().detach().numpy())
        latents.append(embeddings) 
        

        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(loader)
    latents = torch.cat(latents, dim=0)

    return model, train_loss, total_labels, total_preds, latents


def validation (loader, model, loss_fn, task_type = None,
                gene2vec_flag = False, device = "cuda"):
    
    val_loss = 0
    total_preds = []
    total_labels = []

    
    model.eval()
    with torch.no_grad():
        for batch in loader: 

            if gene2vec_flag:
                batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)




            else:
                batch_inputs, batch_masks , labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks)

                
                
            if task_type == "regression":
            
                preds = preds.squeeze().float()
                labels = labels.squeeze().float()

                loss = loss_fn(preds, labels)


                total_preds.extend(preds.cpu().detach())
                total_labels.extend(labels.cpu().detach())        

        
            elif task_type == "classification":

                loss = loss_fn(preds, labels)

                total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
                total_labels.extend(labels.type(torch.int).to('cpu').numpy())

            elif task_type == "multilabel":

                preds = preds.to(torch.float32)
                labels = labels.to(torch.float32)


                loss = loss_fn(preds, labels)

                total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
                total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())

            
            val_loss += loss.item()
                

    val_loss /= len(loader)

    return model, val_loss, total_labels, total_preds


def test(loader, model, loss_fn, task_type = None, gene2vec_flag = False, device = "cuda"):
    
    test_loss = 0
    total_preds = []
    total_labels = []
    latents = []

    
#     if task_type == "regression":
#         loss_fn = nn.MSELoss()
        
#     elif task_type == "classification":
#         loss_fn = nn.CrossEntropyLoss()
            
#     elif task_type == "multilabel":
# #         loss_fn = MultiLabelFocalLoss(alpha=0.25, gamma=2)
#         loss_fn = nn.BCELoss()

#     else:
#         raise ValueError(f"task type errot: {task_type}")
    
    
    model.eval()
    with torch.no_grad():
        for batch in loader: 

            if gene2vec_flag:
                batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)
                


            else:
                batch_inputs, batch_masks, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks)      
            
            
            if task_type == "regression":
            
                preds = preds.squeeze().float()
                labels = labels.squeeze().float()

                loss = loss_fn(preds, labels)
 

                total_preds.extend(preds.cpu().detach())
                total_labels.extend(labels.cpu().detach())        

        
            elif task_type == "classification":

                loss = loss_fn(preds, labels)

                total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
                total_labels.extend(labels.type(torch.int).to('cpu').numpy())

            
            elif task_type == "multilabel":

                preds = preds.to(torch.float32)
                labels = labels.to(torch.float32)


                loss = loss_fn(preds, labels)

                total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
                total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())

            
            test_loss += loss.item()
            

            embeddings = torch.tensor(embeddings.cpu().detach().numpy())
            latents.append(embeddings)

    test_loss /= len(loader)
    latents = torch.cat(latents, dim=0)
       
    return model, test_loss, total_labels, total_preds, latents




def get_metrics(y_true , y_pred, history,  val_type = "Train",
                task_type = "classification"):
    
    
    if task_type == "classification" or task_type == "multilabel":
    
        average = "samples" if task_type == "multilabel" else "weighted"
    
        acc= accuracy_score(y_true , y_pred)
        f1 = f1_score(y_true , y_pred, average=average, zero_division=np.nan)
        prec = precision_score(y_true , y_pred, average=average, zero_division=np.nan)
        rec = recall_score(y_true , y_pred, average=average, zero_division=np.nan)

        history[val_type]["Accuracy"].append(acc)
        history[val_type]["F1"].append(f1)
        history[val_type]["Precision"].append(prec)
        history[val_type]["Recall"].append(rec)
        
        return acc, f1 , prec , rec
        
    else:
        
        
        corrcoef = spearmanr(y_true, y_pred)[0]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)        
        
        history[val_type]["Correlation"].append(corrcoef)
        history[val_type]["MAE"].append(mae)
        history[val_type]["MSE"].append(mse)
        history[val_type]["R2"].append(r2)
        
        return corrcoef, mae, mse, r2


def plot_latent(latents, labels, epoch, class_map = None,
                task_name= "subloc", validation_type="train"):
    
    tsne = TSNE(n_components=2)
    scaler = StandardScaler()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latents_tsne = tsne.fit_transform(latents)
    

    if class_map is not None:
        
        for i, class_label in enumerate(np.unique(labels)):
            class_indices = labels == class_label
            cl = class_map[class_label]
            plt.scatter(latents_tsne[class_indices, 0],
                        latents_tsne[class_indices, 1],
                        label=f'{cl}')
        plt.legend()
        
    else:
        plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1])


def save_finetuned_embeddings(genes, pool = "cls", max_length= 100, batch_size =100, drop_rate =0.1,
                gene2vec_flag = False, gene2vec_hidden = 200, device = "cuda",
                task_type = "classification", n_labels = 3 , model_name= "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract", task_name = 'Subcellular_location'): 

    model = FineTunedBERT(pool= pool, task_type = task_type, n_labels = n_labels,
                          drop_rate = drop_rate, model_name = model_name,
                          gene2vec_flag= gene2vec_flag,
                          gene2vec_hidden = gene2vec_hidden).to(device)
    
    # optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    state_dict = torch.load(f'../../data/{task_name}/best_model_{task_name}.pth')
    model.load_state_dict(state_dict)

    # Tokenize the gene summaries
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    encoded_summaries = tokenizer.batch_encode_plus(genes["Summary"].tolist(), 
                                                   max_length=max_length, 
                                                   padding="max_length",
                                                   truncation=True,
                                                   return_tensors="pt")

    # DataLoader for all genes
    all_dataset = TensorDataset(encoded_summaries["input_ids"], encoded_summaries["attention_mask"])
    all_data_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

    # Store gene names separately
    all_gene_names = genes["Gene name"].tolist()

    # Get embeddings for all genes
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks) in enumerate(all_data_loader):
            embeddings, _, _ = model(inputs.to(device), masks.to(device))
            all_embeddings.append(embeddings.cpu().numpy())

    # Flatten embeddings list
    all_embeddings = np.vstack(all_embeddings)

    
    embeddings_filename = f'../../data/{task_name}/fine_tuned_embeddings_{task_name}.csv'
    embeddings_df = pd.DataFrame(all_embeddings)
    embeddings_df['gene_name'] = all_gene_names  

    
    embeddings_df.to_csv(embeddings_filename, index=False)
 

    embeddings_df = pd.concat([embeddings_df.iloc[:, -1], embeddings_df.iloc[:, :-1]], axis=1)
    embeddings_df.columns = [''] * len(embeddings_df.columns)
    embeddings_filename = f'../../data/{task_name}/fine_tuned_embeddings_{task_name}.csv'
    embeddings_df.to_csv(embeddings_filename, header=False, index=False)
    print(f'Fine-tuned embeddings saved to {embeddings_filename}')
    #print(f'best epoch number: {best_epoch_num}')