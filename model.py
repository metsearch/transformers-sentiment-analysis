import os

import torch as th
import torch.nn as nn

from torchinfo import summary
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    BertTokenizer,
    BertModel
)

from utilities.utils import *

def load_pretrained_model(model_name='distilbert-base-uncased'):
    if model_name == 'distilbert-base-uncased':
        pretrained_model = DistilBertModel.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    else:
        pretrained_model = BertModel.from_pretrained(model_name)
        del pretrained_model.pooler ; pretrained_model.pooler = None
        tokenizer = BertTokenizer.from_pretrained(model_name)

    return pretrained_model, tokenizer

class DistilBert(nn.Module):
    def __init__(self, pretrained_model, latent_size=768, num_classes=2):
        super(DistilBert, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, latent_size, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=latent_size, out_features=num_classes, bias=True),
        )
        
    def forward(self, input_ids=None, attention_mask=None):
        x = self.pretrained_model(input_ids, attention_mask=attention_mask, output_attentions=False)
        x = x[0][:, 0, :]
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    pretrained_model, _ = load_pretrained_model()
    model = DistilBert(pretrained_model).to(device)
    model.train()
    optimzer = th.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    summary(model, depth=4)