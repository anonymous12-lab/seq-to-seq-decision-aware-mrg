import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, RandomSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import BertPreTrainedModel,BertModel
from transformers import BertForSequenceClassification, AdamW, BertConfig



import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig

data=pd.read_csv('decision_data.csv',sep='\t')
data=data.dropna()

import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

sentences1 = data.review1.values
sentences2 = data.review2.values
sentences3 = data.review3.values
labels = data.decision.values

confidence1 = data.C1.values
confidence2 = data.C2.values
confidence3 = data.C3.values

rating1 = data.R1.values
rating2 = data.R2.values
rating3 = data.R3.values

hedge_score1=data.hedge_score1.values
hedge_score2=data.hedge_score2.values
hedge_score3=data.hedge_score3.values

max_len=512


input_ids1 = []
input_ids2 = []
input_ids3 = []
attention_masks1 = []
attention_masks2 = []
attention_masks3 = []
sentiments= []
i=0
for sent1,sent2,sent3 in tqdm(zip(sentences1,sentences2,sentences3),total=len(sentences2)):
    encoded_dict = tokenizer.encode_plus(
                        sent1,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    input_ids1.append(encoded_dict['input_ids'])
    attention_masks1.append(encoded_dict['attention_mask'])
    # sentiments.append(senti)
    encoded_dict = tokenizer.encode_plus(
                        sent2,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

    input_ids2.append(encoded_dict['input_ids'])
    attention_masks2.append(encoded_dict['attention_mask'])
    encoded_dict = tokenizer.encode_plus(
                        sent3,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    input_ids3.append(encoded_dict['input_ids'])
    attention_masks3.append(encoded_dict['attention_mask'])

# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
# rating = torch.tensor(rating)
# confidence=torch.tensor(confidence)


input_ids1 = torch.cat(input_ids1, dim=0)
attention_masks1 = torch.cat(attention_masks1, dim=0)
input_ids2 = torch.cat(input_ids2, dim=0)
attention_masks2 = torch.cat(attention_masks2, dim=0)
input_ids3 = torch.cat(input_ids3, dim=0)
attention_masks3 = torch.cat(attention_masks3, dim=0)
labels = torch.FloatTensor(labels)
rating1 = torch.tensor(rating1)
confidence1=torch.tensor(confidence1)
rating2 = torch.tensor(rating2)
confidence2=torch.tensor(confidence2)
rating3 = torch.tensor(rating3)
confidence3=torch.tensor(confidence3)
hedge_score1=torch.tensor(hedge_score1)
hedge_score2=torch.tensor(hedge_score2)
hedge_score3=torch.tensor(hedge_score3)

print(input_ids1.shape)
print(attention_masks1.shape)
print(labels.shape)

from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(input_ids1, input_ids2, input_ids3, attention_masks1, attention_masks2, attention_masks3, labels)


model = BertModel.from_pretrained('bert-base-uncased')

print(model)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


val_bs=1
validation_dataloader = DataLoader(
            dataset, # The validation samples.
            sampler = SequentialSampler(dataset), # Pull out batches sequentially.
            batch_size = val_bs # Evaluate with this batch size.
        )

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)


num=0
data=[]
for step, batch in tqdm(enumerate(validation_dataloader),total=len(validation_dataloader)):

        b_input_ids1 = batch[0].to(device)
        b_input_ids2 = batch[1].to(device)
        b_input_ids3 = batch[2].to(device)
        b_input_mask1 = batch[3].to(device)
        b_input_mask2 = batch[4].to(device)
        b_input_mask3 = batch[5].to(device)
        b_labels = batch[6].to(device)

        model.zero_grad()        

        with torch.no_grad():
            # print(model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels))
            check1 = model(b_input_ids1, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask1)
            logits1=check1[1]

            check2 = model(b_input_ids2, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask2)
            logits2=check2[1]

            check3 = model(b_input_ids3, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask3)
            logits3=check3[1]

            data.append({'rep1':logits1,'rep2':logits2,'rep3':logits2,
                'review1':sentences1[num],'review2':sentences2[num],'review3':sentences3[num],
                'conbfidence1':confidence1[num],'conbfidence2':confidence2[num],'conbfidence3':confidence3[num],
                'rating1':rating1[num],'rating2':rating2[num],'rating3':rating3[num],
                'hedge_score1':hedge_score1[num],'hedge_score2':hedge_score2[num],'hedge_score3':hedge_score3[num],'labels':labels[num]})
            num+=1

torch.save(data,f'decision_train_pt.pth')
print('data saved')