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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig

data=pd.read_csv('train.csv',sep='\t')
data=data[data.rating!=5]
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
sentences = data.review.values
labels = data.rating.values
sentiment=data.rating.values

len(sentences),len(labels)
max_len = 512

# for sent in sentences:

#     # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
#     input_ids = tokenizer.encode(sent, add_special_tokens=True, max_length=512)

#     max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)
input_ids = []
attention_masks = []
sentiments= []
i=0
for sent,senti in tqdm(zip(sentences,sentiment),total=len(sentences)):
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 75,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    sentiments.append(senti)
    # i+=1
    # if i==400:
    # 	break
    

input_ids = torch.cat(input_ids, dim=0)
print(attention_masks[0].shape)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)
sentiments = torch.tensor(sentiments)
print(sentiments.shape)
print(attention_masks.shape)
print(input_ids.shape)
print('Original: ', sentences[1])
print('Token IDs:', input_ids[1])

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32


from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(input_ids, attention_masks, labels,sentiments)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

val_bs=32
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = val_bs # Evaluate with this batch size.
        )


from transformers import BertPreTrainedModel,BertModel
config = BertConfig.from_pretrained('bert-base-uncased',num_labels=5)
config.output_attentions=False
config.output_hidden_states=True

# class Classification(BertPreTrainedModel):
#     def __init__(self, config):

#         super(Classification, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.batchnorm_class = nn.BatchNorm1d(768)
#         self.relu = nn.ReLU()
#         self.l1=nn.Linear(768,5)
#         self.classifier = nn.Linear(6, 5)
#         self.init_weights()

#     def forward(self, input_ids, sentiment=None,labels=None,token_type_ids=None,attention_mask=None):
#         x = self.relu(self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)[1])
#         # print(x.shape)
#         sentiment=torch.unsqueeze(sentiment, 1)
#         # print(sentiment.shape)
        
#         x = self.batchnorm_class(x)
#         x = self.l1(x)
#         x = torch.cat((x,sentiment),dim=1)
#         x = self.relu(x)
#         logits = self.classifier(x)
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             # Only keep active parts of the loss
#             # attention_mask_label = None
#             # if attention_mask_label is not None:
#             #     active_loss = attention_mask_label.view(-1) == 1
#             #     active_logits = logits.view(-1, self.num_labels)[active_loss]
#             #     active_labels = labels.view(-1)[active_loss]
#             #     loss = loss_fct(active_logits, active_labels)
#             # else:
#             loss = loss_fct(logits.view(-1, self.num_labels),
#                             labels.view(-1))

#             outputs = (loss, logits)
#             return outputs
#         else:
#             return logits    
#         return logits

from transformers import BertForSequenceClassification, AdamW, BertConfig


# p=Classification.from_pretrained('bert-base-uncased',num_labels=5,from_tf=False)


class T(nn.Module):
    def __init__(self):
        super(T, self).__init__()        # Number of input features is 12.
        # self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels = 5,output_attentions = False,output_hidden_states = True)
        self.p1=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=5,from_tf=False)
        self.num_labels = config.num_labels
        
        self.relu = nn.ReLU()
        self.batchnorm_class = nn.BatchNorm1d(5)
        self.dropout = nn.Dropout(p=0.1)
        
        self.classifier = nn.Linear(6,config.num_labels)
        
    def forward(self,input_ids,sentiment=None,labels=None,token_type_ids=None,attention_mask=None):
        x = self.p1(input_ids,labels=None,token_type_ids=token_type_ids,attention_mask=attention_mask)[0]
        x = self.relu(x)
        # x = torch.cat((y,sentiment),dim=1)
        x = self.batchnorm_class(x)
        # print(x.shape,'ssssss')
        
        sentiment=torch.unsqueeze(sentiment, 1)
        # print(sentiment.shape,'nnnnnnn')
        x = torch.cat((x,sentiment),dim=1)
        x = self.relu(x)
        logits = self.classifier(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            # attention_mask_label = None
            # if attention_mask_label is not None:
            #     active_loss = attention_mask_label.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_labels = labels.view(-1)[active_loss]
            #     loss = loss_fct(active_logits, active_labels)
            # else:
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1))

            outputs = (loss, logits)
            return outputs
        else:
            return logits    
        return logits

model=T()
model.to(device)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



optimizer = AdamW(model.parameters(),
                  lr = 5e-5,
                  eps = 1e-8
                )
from transformers import get_linear_schedule_with_warmup
epochs = 2
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
import numpy as np
from sklearn.metrics import classification_report
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # print(classification_report(labels_flat,pred_flat))
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))



import random
import numpy as np

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []

total_t0 = time.time()

epochs=20
avg_val_loss_max=10000000
for epoch_i in range(0, epochs):
    

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0
    model.train()

    for step, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):

        # if step % 40 == 0 and not step == 0:
        #     elapsed = format_time(time.time() - t0)
            
        #     print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        #     # break

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_sentiments=batch[3].to(device)

        model.zero_grad()        


        # print(model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels))
        check = model(b_input_ids,sentiment= b_sentiments,attention_mask=b_input_mask,labels=b_labels)
        loss,logits=check[0],check[1]

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    test_labels=[]
    test_logits=[]
    for batch in tqdm(validation_dataloader):
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_sentiments=batch[3].to(device)

        with torch.no_grad():        

            check1 = model(b_input_ids,sentiment= b_sentiments,
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss,logits=check1[0],check1[1]
            # print(b_labels.shape)
            # print(logits.shape)
            # print(b_input_mask.shape)
            # print(b_input_ids.shape)
            
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        test_labels+=list(label_ids.flatten())
        test_logits+=list(np.argmax(logits, axis=1).flatten())        

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    if avg_val_loss<avg_val_loss_max:
      print('Model Saving')
      torch.save(model.state_dict(),f'expriment_rating_with_senti.pth')
      avg_val_loss_max=avg_val_loss
    # print(test_logits)
    # print(len(test_logits),len(test_labels))
    # print(test_labels)
    print(classification_report(test_labels,test_logits))
    
    validation_time = format_time(time.time() - t0)
  
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))