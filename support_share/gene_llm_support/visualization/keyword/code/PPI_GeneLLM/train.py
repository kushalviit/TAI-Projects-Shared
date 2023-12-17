##############################################
# created on: 11/16/2023
# project: GeneLLM
# author: Kushal
# team: Tumor-AI-Lab
##############################################

from data_process_load import siameseDataset,train_test_val_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torchsummary import summary
import torch
import sys

def train_wrapper(model,refined_ppi_file_name, summary_file_name, num_workers, 
                  batch_size,num_epochs,device,learning_rate,train_percent,
                  model_save_path):
    
    splitter =train_test_val_split(file_name=refined_ppi_file_name,train_percent=train_percent)
    train_genes,val_genes,test_genes = splitter.get_details()
    print(f'length of training {len(train_genes)}')
    print(f'Number of interations per epoch {len(train_genes)/batch_size}')
    if len(train_genes)%batch_size != 0:
        sys.exit('current dataloader is suitable only for multiple of batch_size')
    print(f'length of testing {len(test_genes)}')
    print(f'length of validation {len(val_genes)}')
    train_dataset = siameseDataset(file_name=refined_ppi_file_name,
                                   summary_file_name=summary_file_name,
                                   usable_genes=train_genes,
                                   non_usable_genes=val_genes+test_genes)
    
    val_dataset = siameseDataset(file_name=refined_ppi_file_name,
                                   summary_file_name=summary_file_name,
                                   usable_genes=val_genes,
                                   non_usable_genes=train_genes+test_genes)
    
    trainloader = DataLoader(train_dataset,shuffle = True,
                             num_workers=num_workers,
                             batch_size=batch_size)
    valloader = DataLoader(val_dataset,shuffle = True,
                             num_workers=num_workers,
                             batch_size=batch_size)
    #print(model)
    
    loss_history = []
    counter = []
    criterion = nn.BCELoss()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

    for epoch in range(0,num_epochs):
        model.train()
        epoch_loss = 0
        iteration_number = 0
        iteration_loss= 0
        for i,data in enumerate(trainloader,0):
            gene0,gene1,labels = data
            if device == "cuda":
                gene0, gene1, labels = gene0.to(device), gene1.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(gene0,gene1)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            #loss_history.append(loss.item())
            #counter.append(iteration_number)
            epoch_loss+=loss
            #
            iteration_loss+=loss
            if i%100==0:
                if i!=0:
                    iteration_number +=100
                torch.save({'epoch': epoch,'iteration_number': iteration_number,'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss}, model_save_path)
                if i!=0:
                    iteration_loss = iteration_loss/100
                    print(f'epoch : {epoch}, iteration_number : {iteration_number}, loss : {iteration_loss}')
                    iteration_loss=0
        print(f'epoch : {epoch} epoch loss : {epoch_loss/i}') 
        with torch.no_grad():
            mean_acc =0
            num_val = 0
            model.eval()
            for i,data in enumerate(valloader,0):
                gene0,gene1,labels = data
                gene0, gene1, labels = gene0.to(device), gene1.to(device), labels.to(device)
                outputs = model(gene0,gene1)
                acc =(outputs.round()==labels).float().mean()
                mean_acc += acc
                num_val+=1
            mean_acc = mean_acc/num_val
            print(f'Mean Val acc {mean_acc}')
            before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(" Adam lr %.10f -> %.10f" % ( before_lr, after_lr))

    test_dataset = siameseDataset(file_name=refined_ppi_file_name,
                                  summary_file_name=summary_file_name,
                                  usable_genes=test_genes,
                                  non_usable_genes=val_genes+train_genes)
    
    testloader = DataLoader(test_dataset,shuffle = True,
                            num_workers=num_workers,
                            batch_size=batch_size)
    
    with torch.no_grad():
            mean_acc =0
            num_test = 0
            model.eval()
            for j in range(0,10):
                for i,data in enumerate(testloader,0):
                    gene0,gene1,labels = data
                    gene0, gene1, labels = gene0.to(device), gene1.to(device), labels.to(device)
                    outputs = model(gene0,gene1)
                    acc =(outputs.round()==labels).float().mean()
                    mean_acc += acc
                    num_test+=1
            mean_acc = mean_acc/num_test
            print(f'Mean Test accuracy {mean_acc}')
        




