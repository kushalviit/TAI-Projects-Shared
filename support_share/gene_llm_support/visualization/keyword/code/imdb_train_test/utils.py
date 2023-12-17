from dataset import IMDBDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import *
import torch
import numpy as np

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = IMDBDataset(reviews=df.review.to_numpy(),
                   sentiments=df.sentiment.to_numpy(),
                   tokenizer=tokenizer,
                   max_len=max_len
                   )

  return DataLoader(ds, batch_size=batch_size, num_workers=4)


def create_training_tools(args,model,train_data_loader,device):
    optimizer = AdamW(model.parameters(), lr=args.init_lr, correct_bias=False)
    total_steps = len(train_data_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)
    return optimizer,scheduler,loss_fn

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    sentiments = d["sentiments"].to(device)

    outputs,attentions = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, sentiments)

    correct_predictions += torch.sum(preds == sentiments)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      sentiments = d["sentiments"].to(device)

      outputs,attentions = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, sentiments)

      correct_predictions += torch.sum(preds == sentiments)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def train(args,model,train_data_loader,loss_fn,optimizer,device,scheduler,train_len,val_data_loader,val_len):
  train_a = []
  train_l = []
  val_a = []
  val_l = []
  best_accuracy = 0

  for epoch in range(args.epochs):

    print(f'Epoch {epoch + 1}/{args.epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model,train_data_loader, loss_fn, optimizer, device, scheduler, train_len)

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model,val_data_loader,loss_fn, device, val_len)

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    train_a.append(train_acc)
    train_l.append(train_loss)
    val_a.append(val_acc)
    val_l.append(val_loss)

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), 'best_model_state.bin')
      best_accuracy = val_acc