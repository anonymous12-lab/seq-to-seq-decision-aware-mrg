import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, RandomSampler
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# torch.cuda.empty_cache()
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig
import torch


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import scipy
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score


dataset = torch.load(f'train_conf_pt.pth')


labels_dict={1:0,2:1,3:2,4:3,5:4}

for l in range(len(dataset)):
    dataset[l]['conbfidence']=dataset[l]['conbfidence'].apply_(lambda x: labels_dict[x])

import sys

print(dataset[0]['conbfidence'])

from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(dataset,test_size=0.15, random_state=42)
print(len(dataset))
print(dataset[0])
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




class trainData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        t=(self.X_data[index]['rep'],self.X_data[index]['rating'],torch.tensor(self.X_data[index]['hedge_score'], dtype=torch.long))
        return t , torch.tensor(self.X_data[index]['conbfidence'], dtype=torch.long)
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(train_dataset)
    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        t=(self.X_data[index]['rep'],self.X_data[index]['rating'],torch.tensor(self.X_data[index]['hedge_score'], dtype=torch.long))
        return t
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = testData(test_dataset)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32


train_loader = DataLoader(dataset=train_data,sampler = RandomSampler(train_data), batch_size=batch_size)
test_loader = DataLoader(dataset=test_data,sampler = SequentialSampler(test_data), batch_size=batch_size)


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()        # Number of input features is 12.
        self.layer_1 = nn.Linear(768, 128) 
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(66, 5) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.sft = torch.nn.Softmax(dim=1)
        
    def forward(self, inputs,rating,hedge_score):
        
        inputs=torch.squeeze(inputs, 1)
        rating=torch.unsqueeze(rating,1)
        hedge_score=torch.unsqueeze(hedge_score,1)
        # print(inputs.shape,rating.shape)
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        # print(x.shape)
        x = torch.cat((x,rating,hedge_score),dim=1)
        
        x = self.dropout(x)
        x = self.layer_out(x)
        x=self.sft(x)
        
        return x

model = binaryClassification()
print(model)
criterion =nn.CrossEntropyLoss()
    # return loss(logits, labels.squeeze(1)))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

y_test=[]
for i in test_dataset:
    y_test.append(i['conbfidence'])


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def validate(test_loader,model,device):
    logits_list=[]
    model.eval()
    # y_list=[]
    for i in test_loader:
      # X,y=i['input'],i['target']
      i1=i[0].to(device)
      i2=i[1].to(device)
      i3=i[2].to(device)
      logits=model(i1,i2,i3)
      logits_list.append(logits)
    
    all_new_logits = torch.cat(logits_list)
    # all_new_y = torch.cat(y_list).detach().numpy()

    all_new_logits=torch.argmax(all_new_logits,dim=1).cpu().clone().detach().numpy()

    all_new_logits=np.array(all_new_logits)
    # all_new_y=np.array(all_new_y)
    # print(f"Rmse on val batch : {mean_squared_error(y_test,all_new_logits)}")
    # print(pearsonr(y_test, all_new_logits)[0])
    return f1_score(y_test,all_new_logits,average='macro') #pearsonr(y_test, all_new_logits)

t=0
torch.set_grad_enabled(True)
for e in tqdm(range(1, 100+1)):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch,X_rating,X_hedge, y_batch = X_batch[0].to(device),X_batch[1].to(device),X_batch[2].to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        logits = model(X_batch,X_rating,X_hedge)
        # print(type(logits),type(y_batch))
        loss = criterion(logits, y_batch)
        # acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        # epoch_acc += acc.item()
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    acc_test=validate(test_loader,model,device)
    # print('pearson',acc_test)
    if(e>8):
      if(t<acc_test):
        t=acc_test
        print('Model Saving')
        torch.save(model.state_dict(),f'confi_rating_hedge_best.pth')
     

    # print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')


model.load_state_dict(torch.load(f'confi_rating_hedge_best.pth'))

logits_list=[]
model.eval()
# y_list=[]
for i in test_loader:
  # X,y=i['input'],i['target']
  i1=i[0].to(device)
  i2=i[1].to(device)
  i3=i[2].to(device)
  logits=model(i1,i2,i3)
  logits_list.append(logits)

all_new_logits = torch.cat(logits_list)
# all_new_y = torch.cat(y_list).detach().numpy()

all_new_logits=torch.argmax(all_new_logits,dim=1).cpu().clone().detach().numpy()

all_new_logits=np.array(all_new_logits)
# all_new_y=np.array(all_new_y)
print(f"Rmse on val batch : {mean_squared_error(y_test,all_new_logits, squared=False)}")
print(pearsonr(y_test, all_new_logits))

print(classification_report(y_test, all_new_logits))