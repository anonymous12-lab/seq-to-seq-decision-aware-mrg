import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, RandomSampler
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="3"
torch.cuda.empty_cache()
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
import itertools as it


dataset = torch.load(f'decision_train_pt.pth')
from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(dataset,test_size=0.15, random_state=42)
print(len(dataset))
print(dataset[0])


from nltk.sentiment.vader import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()

def Vader(review):
    polarity = sid.polarity_scores(review)
    sorted_keys = sorted(polarity.keys())
    return [polarity[k] for k in sorted_keys]

def sentiment( x):  #input list of lists(tokenized sentences for each review)
    # print('No. of Reviews: {}'.format(len(x)))
    lens = list(map(lambda i:len(i), x))
    x = list(it.chain.from_iterable(x))   #list of sentences
    # print(x)
    # print('No. of Sentences: {}'.format(len(x)))
    sentic = []
    zero = [0.0]*4
    ir = iter(x)
    for i in lens:
        score = []
        while len(score) < i:
            score.append(Vader(next(ir)))
        sentic.append(score)
    sentic = np.array(list(zip(*list(it.zip_longest(*sentic, fillvalue = zero)))))
    return sentic

q=np.zeros((200,), dtype=float)

print(np.array(sentiment([dataset[0]['review1'].split('.')])).shape)
p=np.array(sentiment([dataset[0]['review1'].split('.')])).reshape(64)
print(p)

for i in range(p.shape[0]):
    q[i]=p[i]
print(q)


import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
z=tokenizer.tokenize(dataset[0]['review1'])
# print(z)
# print(dataset[0]['review'].split('.'))


def cal_senti(X_data):
    for index in tqdm(range(len(X_data))):
        review=X_data[index]['review1']
        z=tokenizer.tokenize(review)[:15]
        senti=sentiment([z])
        # print(senti)

        senti=senti.reshape(senti.shape[1]*4)
        # print(senti.shape,'ssssssssssss')
        q=np.zeros((60,), dtype=float)
        # print(q.shape)
        for t in range(len(senti)):
            q[t]=senti[t]
        X_data[index]['sentiment1']=q

    for index in tqdm(range(len(X_data))):
        review=X_data[index]['review2']
        z=tokenizer.tokenize(review)[:15]
        senti=sentiment([z])
        # print(senti)
        # print(senti.shape,'ssssssssssss')
        senti=senti.reshape(senti.shape[1]*4)
        q=np.zeros((60,), dtype=float)
        for t in range(len(senti)):
            q[t]=senti[t]
        X_data[index]['sentiment2']=q

    for index in tqdm(range(len(X_data))):
        review=X_data[index]['review3']
        z=tokenizer.tokenize(review)[:15]
        senti=sentiment([z])
        # print(senti)
        # print(senti.shape,'ssssssssssss')
        senti=senti.reshape(senti.shape[1]*4)
        q=np.zeros((60,), dtype=float)
        for t in range(len(senti)):
            q[t]=senti[t]
        X_data[index]['sentiment3']=q
    return X_data

dataset=cal_senti(dataset)

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



# class trainData(Dataset):
    
#     def __init__(self, X_data):
#         self.X_data = X_data
        
#     def __getitem__(self, index):
#         review=self.X_data[index]['review']
#         z=tokenizer.tokenize(review)[:50]
#         senti=sentiment([z])
#         # print(senti)
#         # print(senti.shape,'ssssssssssss')
#         senti=senti.reshape(senti.shape[1]*4)
#         q=np.zeros((200,), dtype=float)
#         for t in range(len(senti)):
#             q[t]=senti[t]
#         return self.X_data[index]['rep'], torch.FloatTensor(q), self.X_data[index]['rating']
        
#     def __len__ (self):
#         return len(self.X_data)


# train_data = trainData(train_dataset)
    
# class testData(Dataset):
    
#     def __init__(self, X_data):
#         self.X_data = X_data
        
#     def __getitem__(self, index):
#         review=self.X_data[index]['review']
#         z=tokenizer.tokenize(review)[:50]
#         senti=sentiment([z])
#         # print(senti)
#         # print(senti.shape,'ssssssssssss')
#         senti=senti.reshape(senti.shape[1]*4)
#         q=np.zeros((200,), dtype=float)
#         for t in range(len(senti)):
#             q[t]=senti[t]
#         return self.X_data[index]['rep'], torch.FloatTensor(q)
        
#     def __len__ (self):
#         return len(self.X_data)
    

# test_data = testData(test_dataset)

class trainData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        t=(self.X_data[index]['rep1'],self.X_data[index]['rep2'],self.X_data[index]['rep3'],
            self.X_data[index]['rating1'],self.X_data[index]['rating2'],self.X_data[index]['rating3'],
            self.X_data[index]['conbfidence1'],self.X_data[index]['conbfidence2'],self.X_data[index]['conbfidence3'],
            torch.tensor(self.X_data[index]['hedge_score1'], dtype=torch.long),torch.tensor(self.X_data[index]['hedge_score2'], dtype=torch.long),
            torch.tensor(self.X_data[index]['hedge_score3'], dtype=torch.long),
            torch.FloatTensor(self.X_data[index]['sentiment1']),torch.FloatTensor(self.X_data[index]['sentiment2']),
            torch.FloatTensor(self.X_data[index]['sentiment3']))

        return t, self.X_data[index]['labels']
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(train_dataset)
    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        t=(self.X_data[index]['rep1'],self.X_data[index]['rep2'],self.X_data[index]['rep3'],
            self.X_data[index]['rating1'],self.X_data[index]['rating2'],self.X_data[index]['rating3'],
            self.X_data[index]['conbfidence1'],self.X_data[index]['conbfidence2'],self.X_data[index]['conbfidence3'],
            torch.tensor(self.X_data[index]['hedge_score1'], dtype=torch.long),torch.tensor(self.X_data[index]['hedge_score2'], dtype=torch.long),
            torch.tensor(self.X_data[index]['hedge_score3'], dtype=torch.long),
            torch.FloatTensor(self.X_data[index]['sentiment1']),torch.FloatTensor(self.X_data[index]['sentiment2']),
            torch.FloatTensor(self.X_data[index]['sentiment3']))
        return t
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = testData(test_dataset)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32


train_loader = DataLoader(dataset=train_data,sampler = RandomSampler(train_data), batch_size=batch_size)
test_loader = DataLoader(dataset=test_data,sampler = SequentialSampler(test_data), batch_size=batch_size)


# class binaryClassification(nn.Module):
#     def __init__(self):
#         super(binaryClassification, self).__init__()        # Number of input features is 12.
#         self.layer_1 = nn.Linear(768, 128) 
#         self.layer_2 = nn.Linear(128, 64)
#         self.layer_out = nn.Linear(64, 5)
#         self.sentiment=nn.Linear(200,64)
#         self.batchnorm_s=nn.BatchNorm1d(64)
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.1)
#         self.batchnorm1 = nn.BatchNorm1d(128)
#         self.batchnorm2 = nn.BatchNorm1d(64)
#         self.sft = torch.nn.Softmax(dim=1)
        
#     def forward(self, inputs,sentiment):
#         inputs=torch.squeeze(inputs, 1)
#         x = self.relu(self.layer_1(inputs))
#         s = self.sentiment(sentiment)
#         s = self.batchnorm_s(s)
#         x = self.batchnorm1(x)
#         x = self.relu(self.layer_2(x)+s)
#         # print(x.shape)
#         x = self.batchnorm2(x)
#         x = self.dropout(x)
#         x = self.layer_out(x)
#         x=self.sft(x)
        
#         return x

# class binaryClassification(nn.Module):
#     def __init__(self):
#         super(binaryClassification, self).__init__()        # Number of input features is 12.
#         self.layer_1_1 = nn.Linear(768, 192)
#         self.layer_1_2 = nn.Linear(768, 192)
#         self.layer_1_3 = nn.Linear(768, 192)
#         self.layer_2 = nn.Linear(774, 768)
#         # self.layer_2_2 = nn.Linear(128, 64)
#         # self.layer_2_3 = nn.Linear(128, 64)
#         self.layer_out = nn.Linear(768, 1)
#         self.sentiment1=nn.Linear(200,64)
#         self.sentiment2=nn.Linear(200,64)
#         self.sentiment3=nn.Linear(200,64)

#         self.batchnorm1_1 = nn.BatchNorm1d(192)
#         self.batchnorm1_2 = nn.BatchNorm1d(192)
#         self.batchnorm1_3 = nn.BatchNorm1d(192)

#         self.batchnorm_s1=nn.BatchNorm1d(64)
#         self.batchnorm_s2=nn.BatchNorm1d(64)
#         self.batchnorm_s3=nn.BatchNorm1d(64)

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.1)
#         self.batchnorm1 = nn.BatchNorm1d(128)
#         self.batchnorm2 = nn.BatchNorm1d(768)
#         # self.sft = torch.nn.Softmax(dim=1)
        
#     def forward(self, inputs1,inputs2,inputs3,sentiment1,sentiment2,sentiment3):
#         inputs1_=torch.squeeze(inputs1[0], 1)
#         inputs2_=torch.squeeze(inputs2[0], 1)
#         inputs3_=torch.squeeze(inputs3[0], 1)
        
#         rating1=torch.unsqueeze(inputs1[1],1)
#         rating2=torch.unsqueeze(inputs2[1],1)
#         rating3=torch.unsqueeze(inputs3[1],1)
        
#         conf1=torch.unsqueeze(inputs1[2],1)
#         conf2=torch.unsqueeze(inputs2[2],1)
#         conf3=torch.unsqueeze(inputs3[2],1)

        
#         # rating1=inputs1[1]
#         # rating2=inputs2[1]
#         # rating3=inputs3[1]
        
#         # conf1=inputs1[2]
#         # conf2=inputs2[2]
#         # conf3=inputs3[2] 

#         x1 = self.relu(self.layer_1_1(inputs1_))
#         x1 = self.batchnorm1_1(x1)
#         x2 = self.relu(self.layer_1_2(inputs2_))
#         x2 = self.batchnorm1_2(x2)
#         x3 = self.relu(self.layer_1_3(inputs3_))
#         x3 = self.batchnorm1_3(x3)

#         s1 = self.sentiment1(sentiment1)
#         s1 = self.batchnorm_s1(s1)

#         s2 = self.sentiment2(sentiment2)
#         s2 = self.batchnorm_s2(s2)

#         s3 = self.sentiment3(sentiment3)
#         s3 = self.batchnorm_s3(s3)

#         # x = self.batchnorm1(x)
#         x = torch.cat((x1,s1,rating1,conf1,x2,s2,rating1,conf2,x3,s3,rating1,conf3),dim=1)
#         x = self.relu(self.layer_2(x))
        
#         # print(x.shape)
#         x = self.batchnorm2(x)
#         # x = torch.cat((x,s),dim=1)
#         x = self.dropout(x)
#         x = self.layer_out(x)
#         # x=self.sft(x)
        
#         return x



class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()        # Number of input features is 12.
        self.layer_1_1 = nn.Linear(2304, 768)
        self.layer_2 = nn.Linear(789, 768)
        # self.layer_2_2 = nn.Linear(128, 64)
        # self.layer_2_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(768, 1)
        self.sentiment1=nn.Linear(60,4)
        self.sentiment2=nn.Linear(60,4)
        self.sentiment3=nn.Linear(60,4)
        # self.s=nn.Linear(198,32)

        self.batchnorm1_1 = nn.BatchNorm1d(768)
        self.batchnorm1_2 = nn.BatchNorm1d(192)
        self.batchnorm1_3 = nn.BatchNorm1d(192)

        self.batchnorm_s1=nn.BatchNorm1d(4)
        self.batchnorm_s2=nn.BatchNorm1d(4)
        self.batchnorm_s3=nn.BatchNorm1d(4)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(18)
        self.batchnorm2 = nn.BatchNorm1d(768)
        # self.sft = torch.nn.Softmax(dim=1)
        
    def forward(self, inputs1,inputs2,inputs3,sentiment1,sentiment2,sentiment3):
        inputs1_=torch.squeeze(inputs1[0], 1)
        inputs2_=torch.squeeze(inputs2[0], 1)
        inputs3_=torch.squeeze(inputs3[0], 1)
        
        rating1=torch.unsqueeze(inputs1[1],1)
        rating2=torch.unsqueeze(inputs2[1],1)
        rating3=torch.unsqueeze(inputs3[1],1)
        
        conf1=torch.unsqueeze(inputs1[2],1)
        conf2=torch.unsqueeze(inputs2[2],1)
        conf3=torch.unsqueeze(inputs3[2],1)


        hedge1=torch.unsqueeze(inputs1[3],1)
        hedge2=torch.unsqueeze(inputs2[3],1)
        hedge3=torch.unsqueeze(inputs3[3],1)

        x1 = self.relu(self.layer_1_1(torch.cat((inputs1_,inputs2_,inputs3_),dim=1)))
        x = self.batchnorm1_1(x1)

        s1 = self.sentiment1(sentiment1)
        s1 = self.batchnorm_s1(s1)
        s1=self.relu(s1)

        s2 = self.sentiment2(sentiment2)
        s2 = self.batchnorm_s2(s2)
        s2=self.relu(s2)

        s3 = self.sentiment3(sentiment3)
        s3 = self.batchnorm_s3(s3)
        s3=self.relu(s3)

        # x = self.batchnorm1(x)
        senti = torch.cat((s1,s2,s3,conf1,conf2,conf3,hedge1,hedge2,hedge3,rating1,rating2,rating3),dim=1)
        # senti=self.s(senti)
        # x = self.batchnorm1(x)
        x=torch.cat((x1,senti),dim=1)
        x = self.relu(self.layer_2(x))
        
        # print(x.shape)
        x = self.batchnorm2(x)
        # x = torch.cat((x,s),dim=1)
        x = self.dropout(x)
        x = self.layer_out(x)
        # x=self.sft(x)
        
        return x

model = binaryClassification()
print(model)
criterion =nn.BCEWithLogitsLoss()
    # return loss(logits, labels.squeeze(1)))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

y_test=[]
for i in test_dataset:
    y_test.append(i['labels'])



def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def validate(test_loader,model,device):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_input_b1=X_batch[0].to(device)
            X_input_b2=X_batch[1].to(device)
            X_input_b3=X_batch[2].to(device)
            
            X_rating_b1=X_batch[3].to(device)
            X_rating_b2=X_batch[4].to(device)
            X_rating_b3=X_batch[5].to(device)
            
            X_confi_b1=X_batch[6].to(device)
            X_confi_b2=X_batch[7].to(device)
            X_confi_b3=X_batch[8].to(device)

            X_senti_b1=X_batch[12].to(device)
            X_senti_b2=X_batch[13].to(device)
            X_senti_b3=X_batch[14].to(device)

            X_hedge_b1=X_batch[9].to(device)
            X_hedge_b2=X_batch[10].to(device)
            X_hedge_b3=X_batch[11].to(device)

            y_test_pred = model([X_input_b1,X_rating_b1,X_confi_b1,X_hedge_b1],
                [X_input_b2,X_rating_b2,X_confi_b2,X_hedge_b2],
                [X_input_b3,X_rating_b3,X_confi_b3,X_hedge_b3],
                X_senti_b1,X_senti_b2,X_senti_b3)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    # print(accuracy_score(y_test, y_pred_list))
    return accuracy_score(y_test, y_pred_list)

t=0
torch.set_grad_enabled(True)
for e in tqdm(range(1, 100+1)):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_input_b1=X_batch[0].to(device)
        X_input_b2=X_batch[1].to(device)
        X_input_b3=X_batch[2].to(device)
        
        X_rating_b1=X_batch[3].to(device)
        X_rating_b2=X_batch[4].to(device)
        X_rating_b3=X_batch[5].to(device)
        
        X_confi_b1=X_batch[6].to(device)
        X_confi_b2=X_batch[7].to(device)
        X_confi_b3=X_batch[8].to(device)

        X_senti_b1=X_batch[12].to(device)
        X_senti_b2=X_batch[13].to(device)
        X_senti_b3=X_batch[14].to(device)

        X_hedge_b1=X_batch[9].to(device)
        X_hedge_b2=X_batch[10].to(device)
        X_hedge_b3=X_batch[11].to(device)
        
        y_batch=y_batch.to(device)
        
        optimizer.zero_grad()
        
        logits = model([X_input_b1,X_rating_b1,X_confi_b1,X_hedge_b1],
            [X_input_b2,X_rating_b2,X_confi_b2,X_hedge_b2],
            [X_input_b3,X_rating_b3,X_confi_b3,X_hedge_b3],
            X_senti_b1,X_senti_b2,X_senti_b3)
        # print(type(logits),type(y_batch))
        loss = criterion(logits, y_batch.unsqueeze(1))
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
        print(acc_test)
        print('Model Saving')
        torch.save(model.state_dict(),f'decision_senti_rating_conf_hedge_best.pth')
     

    # print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

model.load_state_dict(torch.load(f'decision_senti_rating_conf_hedge_best.pth'))

y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_input_b1=X_batch[0].to(device)
        X_input_b2=X_batch[1].to(device)
        X_input_b3=X_batch[2].to(device)
        
        X_rating_b1=X_batch[3].to(device)
        X_rating_b2=X_batch[4].to(device)
        X_rating_b3=X_batch[5].to(device)
        
        X_confi_b1=X_batch[6].to(device)
        X_confi_b2=X_batch[7].to(device)
        X_confi_b3=X_batch[8].to(device)

        X_senti_b1=X_batch[12].to(device)
        X_senti_b2=X_batch[13].to(device)
        X_senti_b3=X_batch[14].to(device)

        X_hedge_b1=X_batch[9].to(device)
        X_hedge_b2=X_batch[10].to(device)
        X_hedge_b3=X_batch[11].to(device)

        y_test_pred = model([X_input_b1,X_rating_b1,X_confi_b1,X_hedge_b1],
            [X_input_b2,X_rating_b2,X_confi_b2,X_hedge_b2],
            [X_input_b3,X_rating_b3,X_confi_b3,X_hedge_b3],
            X_senti_b1,X_senti_b2,X_senti_b3)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# all_new_logits = torch.cat(logits_list)
# # all_new_y = torch.cat(y_list).detach().numpy()

# all_new_logits=torch.argmax(all_new_logits,dim=1).cpu().clone().detach().numpy()

# all_new_logits=np.array(all_new_logits)
# # all_new_y=np.array(all_new_y)
# print(f"Rmse on val batch : {mean_squared_error(y_test,all_new_logits)}")
# print(pearsonr(y_test, all_new_logits))

# print(classification_report(y_test, all_new_logits))

print(accuracy_score(y_test, y_pred_list))
print(classification_report(y_test, y_pred_list))