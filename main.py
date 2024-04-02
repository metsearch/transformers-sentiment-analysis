import os
import pickle

import click
import pandas as pd
from datasets import load_dataset

from rich.progress import track

import torch as th
from torch.utils.data import DataLoader

from model import load_pretrained_model
from dataset import IMDBDataset, collate_fn
from model import DistilBert
from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug mode flag', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug 
    invoked_subcommand = ctx.invoked_subcommand 
    if invoked_subcommand is None:
        logger.debug('no subcommand were called')
    else:
        logger.debug(f'{invoked_subcommand} was called')

@router_cmd.command()
@click.option('--path2data', help='where the dataset will be saved', type=str, default='data/')
@click.option('--path2models', help='where the models will be saved', type=str, default='models/')
def grabber(path2data, path2models):
    if not os.path.exists(path2data):
        os.makedirs(path2data)
    if not os.path.exists(path2models):
        os.makedirs(path2models)
    
    pretrained_model, tokenizer = load_pretrained_model()
    dataset = load_dataset('imdb')
    train_data = dataset['train']
    test_data = dataset['test']
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    train_dataset = IMDBDataset(train_df, tokenizer)
    test_dataset = IMDBDataset(test_df, tokenizer)
    
    with open(f'{path2data}/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(f'{path2data}/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)
    with open(f'{path2models}/pretrained_model.pkl', 'wb') as f:
        pickle.dump(pretrained_model, f)
    with open(f'{path2models}/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    logger.info('Dataset and model successfully saved ...!')

@router_cmd.command()
@click.option('--path2data', help='where the dataset is saved', type=str, default='data/')
@click.option('--path2models', help='where the models are saved', type=str, default='models/')
@click.option('--batch_size', help='batch size', type=int, default=32)
@click.option('--num_epochs', help='number of epochs', type=int, default=2)
def train(path2data, path2models, batch_size, num_epochs):
    logger.info('Training ...')
    
    with open(f'{path2data}/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    with open(f'{path2models}/pretrained_model.pkl', 'rb') as f:
        pretrained_model = pickle.load(f)
    model = DistilBert(pretrained_model).to(device)

    optimizer = th.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = th.nn.CrossEntropyLoss()
    
    losses = []
    nb_data = len(train_loader)
    for epoch in range(num_epochs):
        counter = 0
        epoch_loss = 0.0
        model.train()
        for input_ids, attention_mask, labels in track(train_loader, description='Training'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            E: th.Tensor = criterion(outputs, labels)
            E.backward()
            optimizer.step()
            
            epoch_loss += E.cpu().item()
            counter += len(input_ids)
        
        average_loss = epoch_loss / nb_data
        losses.append(average_loss)
        logger.debug(f'[{epoch:03d}/{num_epochs:03d}] [{counter:05d}/{nb_data:05d}] >> Loss : {average_loss:07.3f}')

    th.save(model.cpu(), os.path.join(path2models, 'distilbert.pth'))
    logger.info('The model was saved ...!')

@router_cmd.command()
@click.option('--path2data', help='where the dataset is saved', type=str, default='data/')
@click.option('--path2models', help='where the models are saved', type=str, default='models/')
def inference(path2data, path2models):
    logger.info('Inference ...')
    pass

if __name__ == '__main__':
    router_cmd(obj={})