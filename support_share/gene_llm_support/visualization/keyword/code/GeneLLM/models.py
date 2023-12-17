from transformers import XLNetModel
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class FineTunedBERT(nn.Module):

    def __init__(self, pool="mean", model_name= "bert-base-microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
                 task_type = None, n_labels = None, drop_rate = None,
                 gene2vec_flag=False, gene2vec_hidden = 200):
        
        """
            task_type : regression or classification.
        
        """
      
        super(FineTunedBERT, self).__init__()
        
        assert (task_type == 'regression' and n_labels == 1) or (task_type == 'classification' and n_labels>1) or (task_type == 'multilabel' and n_labels>1), \
            f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"  
        
        # assert gene2vec_flag is not None, f"gene2vec_flag cannot be None: {gene2vec_flag}"

        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "xlnet" in model_name:
            self.bert = XLNetModel.from_pretrained(model_name)
        
        else:
            self.bert = AutoModel.from_pretrained(model_name)
            
    
        self.pool = pool
        bert_hidden = self.bert.config.hidden_size
                
        
        if task_type.lower() == "classification":
            
            if gene2vec_flag:
                self.pipeline = nn.Sequential(nn.Dropout(drop_rate),
                    nn.Linear(bert_hidden+gene2vec_hidden, n_labels)
                )
                
            else:
                self.pipeline = nn.Sequential(nn.Dropout(drop_rate),
                    nn.Linear(bert_hidden, n_labels)
                )

    
        elif task_type.lower() == "multilabel":
            
            if gene2vec_flag:
                self.pipeline = nn.Sequential(nn.Dropout(drop_rate),
                    nn.Linear(bert_hidden+gene2vec_hidden, n_labels),
                    nn.Sigmoid()
                )
                
            else:
                self.pipeline = nn.Sequential(nn.Dropout(drop_rate),
                    nn.Linear(bert_hidden, n_labels),
                    nn.Sigmoid()
                )

                
        elif task_type.lower() == "regression":

            if gene2vec_flag:
                self.pipeline = nn.Sequential(nn.Dropout(drop_rate),
                nn.Linear(bert_hidden+gene2vec_hidden, 1))
                
            else:            
                self.pipeline = nn.Sequential(nn.Dropout(drop_rate),
                nn.Linear(bert_hidden, 1)
                )

        else:
            raise ValueError(f"Key Error task_type : {task_type} ")

            
        
        
    def forward(self, input_ids_, attention_mask_, gene2vec=None):
        
        
        # retrieving the hidden state embeddings
        if "xlnet" in self.model_name:
            output = self.bert(input_ids = input_ids_,
                               attention_mask=attention_mask_)

            hiddenState, ClsPooled = output.last_hidden_state, output.last_hidden_state[:,0, :]
            hiddenState, ClsPooled = hiddenState, ClsPooled


        else:
            hiddenState, ClsPooled = self.bert(input_ids = input_ids_,
                                               attention_mask=attention_mask_).values()

            
        # perform pooling on the hidden state embeddings
        if self.pool.lower() == "max":
            embeddings = self.max_pooling(hiddenState, attention_mask_)
            
        elif self.pool.lower() == "cls":
            embeddings = ClsPooled
                
        elif self.pool.lower() == "mean":
            embeddings = self.mean_pooling(hiddenState, attention_mask_)

        else:
            raise ValueError('Pooling value error.')
        
        
        if gene2vec is not None:
            embeddings = torch.cat((embeddings, gene2vec), dim=1)
      

        return embeddings, hiddenState, self.pipeline(embeddings)

    def max_pooling(self, hidden_state, attention_mask):
        
        #CLS: First element of model_output contains all token embeddings
        token_embeddings = hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        
        pooled_embeddings = torch.max(token_embeddings, 1)[0]
        return pooled_embeddings
    
    def mean_pooling (self, hidden_state, attention_mask):
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) 
        
        return pooled_embeddings

    def get_bert(self):
        return self.bert

    def assign_bert_to_device(self):
        self.bert.to(self.device)

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        
        self.alpha = nn.Parameter(torch.tensor(0.25, requires_grad=True, device="cuda"))  
        self.gamma = nn.Parameter(torch.tensor(2.0, requires_grad=True, device="cuda"))  

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss.mean()


def Logistic_Regression(task_name, finetuned_emb_path, input_data_path):


    df_ft = pd.read_csv(finetuned_emb_path)
    df_ft_label = pd.read_csv(input_data_path)

    df_ft.columns = ['GeneSymbol'] + [f'Col_{i}' for i in range(1, len(df_ft.columns))]
    merged_data = pd.merge(df_ft_label, df_ft, on='GeneSymbol', how='left').dropna()
    # import sys



    # Assuming merged_data is loaded beforehand
    X = merged_data.iloc[:, 2:]
    y = merged_data.iloc[:, 1:2]

    # sys.exit("Exiting the code at a specific point.")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the model with the best parameters
    model = LogisticRegression(C=1, max_iter=1000)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict the test set results
    y_pred = model.predict(X_test)

    joblib.dump(model, f"../../data/{task_name}/logistic_regression_model_{task_name}.pkl")

    # Print the accuracy
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Print the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix: \n {conf_mat}")

    # Print the classification report
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix', size=15)
    plt.savefig(f'../../results/{task_name}/confusion_matrix_LR.png')

    # Calculate the probabilities
    probabilities = model.predict_proba(X_test)

    # Create a label encoder instance
    label_encoder = LabelEncoder()

    # Fit the encoder on the entire target data
    label_encoder.fit(y)

    # Transform y_test and y_pred to binary or multi-class format
    encoded_y_test = label_encoder.transform(y_test)

    # Check the number of unique labels
    # unique_values = y['Solubility'].unique()
    unique_labels = y[f'{y.columns[0]}'].unique()

    if len(unique_labels) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(encoded_y_test, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'../../results/{task_name}/ROC_AUC_CURVE_LR.png')


    else:
        # Multi-class classification
        # One-vs-all ROC AUC curves
        n_classes = len(unique_labels)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((encoded_y_test == i).astype(int), probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'../../results/{task_name}/ROC_AUC_CURVE_LR.png')
