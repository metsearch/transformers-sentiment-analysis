import torch as th
import pandas as pd
from torch.utils.data import Dataset

from utilities.utils import *

class IMDBDataset(Dataset):
    def __init__(self, df:pd.DataFrame, tokenizer, max_length=512):
        self.df = df
        self.max_length = max_length
        self.ds = [None] * self.df.shape[0]
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        temp = self.ds[idx]
        if temp is None:
            sample = self.df.iloc[idx]
            result = self.tokenizer(sample['text'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            temp = (
                result['input_ids'],
                result['attention_mask'],
                th.tensor(sample['label'])
            )
            self.ds[idx] = temp
            return temp
        else:
            return temp

def collate_fn(samples):
    input_ids, attention_mask, labels = list(zip(*samples))
    return (
        th.cat(input_ids, dim=0),
        th.cat(attention_mask, dim=0),
        th.stack(labels, dim=0)
    )   

if __name__ == '__main__':
    logger.info('Testing dataset...')