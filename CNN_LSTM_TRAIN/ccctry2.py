#coding:utf-8
"""
@author: lsx
"""
import argparse
import datetime
import logging
import sys
from pathlib import Path
import numpy as np
import copy
from tqdm import tqdm
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from classify import load_imglist
#from torch.utils.tensorboard import SummaryWriter
from ccctry import make_train_dataloader
#from dataLoad import make_train_dataloader
from CNNLSTM import *
#from ccctry import *
import torchvision
import cv2 
import torch.nn.functional as func
#import iterater as ite
import math
#from model import SegNet
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class lstm_cell(nn.Module):
    def __init__(self, input_num, hidden_num):
        super(lstm_cell, self).__init__()

        self.input_num = input_num
        self.hidden_num = hidden_num

        self.Wxi = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whi = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxf = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whf = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxc = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whc = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxo = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Who = nn.Linear(self.hidden_num, self.hidden_num, bias=False)

    def forward(self, xt, ht_1, ct_1):        
        it = torch.sigmoid(self.Wxi(xt) + self.Whi(ht_1))        
        ft = torch.sigmoid(self.Wxf(xt) + self.Whf(ht_1))        
        ot = torch.sigmoid(self.Wxo(xt) + self.Who(ht_1))        
        ct = ft * ct_1 + it * torch.tanh(self.Wxc(xt) + self.Whc(ht_1))        
        ht = ot * torch.tanh(ct)
        
        return  ht, ct

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))
    

def my_cross_loss(l_t_i, y_t_i, ypre_t_i,lam):
    N = ypre_t_i.size(0)
    for i in range(5):
        for j in range(16):
            for k in range(N):
                if ypre_t_i[k][j][i] < 1e-10:
                    ypre_t_i[k][j][i] = 1e-10
    f1 = -1*torch.sum( torch.sum( torch.mul(y_t_i,torch.log(ypre_t_i)),2),1)    
    T_l = torch.sum(l_t_i,1) 
    f2 = lam*torch.sum((1-T_l)**2,1)
    loss = f1+f2
    N = ypre_t_i.size(0)
    
    loss = torch.mean(loss)
    output = ypre_t_i[:,-1,:]
    prediction = torch.argmax(output, 1)
    print (prediction)
    label = torch.argmax(y_t_i[:,-1,:],1)
    print (label)
    
    for e,la in enumerate(label):
        if la.item()==3 or la.item()==4:
            loss*=2          
    
    acc = 0
    for e in range(N):
        if int(label[e].item())==int(prediction[e].item()):
            acc = acc+1
    acc = acc/(N*1.0)
    return loss,acc



class ALSTM(nn.Module):

    def __init__(self, input_num, hidden_num, num_layers,out_num ):
        
        super(ALSTM, self).__init__()

        # Make sure that `hidden_num` are lists having len == num_layers
        hidden_num = self._extend_for_multilayer(hidden_num, num_layers)
        if not len(hidden_num) == num_layers:
            raise ValueError('The length of hidden_num is not consistent with num_layers.')

        self.input_num = input_num
        self.hidden_num = hidden_num
        self.num_layers = num_layers
        self.out_num = out_num

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_num = self.input_num if i == 0 else self.hidden_num[i - 1]            
            cell_list.append(lstm_cell(cur_input_num,self.hidden_num[i]).cuda())           

        self.cell_list = nn.ModuleList(cell_list)
        #self.conv=nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
        vgg = torchvision.models.vgg16(pretrained=True)
        self.conv=nn.Sequential(*list(vgg.features._modules.values())[:31])
        
        
            
        self.Wha=nn.Linear(self.hidden_num[-1],49)
        self.fc=nn.Linear(self.hidden_num[-1],self.out_num)
        self.softmax=nn.Softmax(dim=1)
        self.tanh=nn.Tanh()
        self.soft_out = nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(p=0.5)
        
    def forward(self, x, hidden_state=None):
        
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=x.size(0))
        out_list=[]
        seq_len = x.size(1)#30
        l_list = []
        for t in range(seq_len):
            output_t = []
            for layer_idx in range(self.num_layers):
                if 0==t:
                    ht_1, ct_1 = hidden_state[layer_idx][0],hidden_state[layer_idx][1].cuda()
                    attention_h=hidden_state[-1][0].cuda()
                else:
                    ht_1, ct_1 = hct_1[layer_idx][0].cuda(),hct_1[layer_idx][1].cuda()
                if 0==layer_idx:
                    feature_map=self.conv(x[:, t, :, :, :]).cuda()
                    feature_map=feature_map.view(feature_map.size(0),feature_map.size(1),-1).cuda()
                    attention_map=self.Wha(attention_h).cuda()
                    attention_map=torch.unsqueeze(self.softmax(attention_map),1).cuda()
                    
                    attention_feature=attention_map.cuda()*feature_map.cuda()
                    attention_feature=torch.sum(attention_feature,2).cuda()
                    
                    ht, ct = self.cell_list[layer_idx](attention_feature.cuda(),ht_1.cuda(), ct_1.cuda())
                    output_t.append([ht.cuda(),ct.cuda()])
                else:
                    ht, ct = self.cell_list[layer_idx](output_t[layer_idx-1][0].cuda(), ht_1.cuda(), ct_1.cuda())
                    output_t.append([ht.cuda(),ct.cuda()])
            attention_h=output_t[-1][0].cuda()
            hct_1=output_t
            
            aaa = self.fc(output_t[-1][0]).cuda()
            aaa = torch.unsqueeze(aaa,0)
            bn = nn.BatchNorm1d(x.size(0))
            bn = bn.cuda()
            bbb = bn(aaa)
            bbb = torch.squeeze(bbb,0)
            out_list.append(self.soft_out(self.tanh(bbb)))  
            l_list.append(attention_map.cuda())

        return torch.stack(out_list,1),torch.stack(l_list,1).squeeze(2)


    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):            
            tensor1 = torch.empty(batch_size, self.hidden_num[i])
            tensor2 = torch.empty(batch_size, self.hidden_num[i])
            ts1 = nn.init.orthogonal_(tensor1)
            ts2 = nn.init.orthogonal_(tensor2)
            
            init_states.append([ts1,ts2])
        return init_states


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param 



def train_batch():
    epochs = 10
    learning_rate = 1e-3 
    image_shape = (3, 224, 224)
    num_classes = 3
    latent_dim =  512
    lstm_layers =  3
    hidden_dim =  1024
    bidirectional =  True
    attention = True
    train_size = 0.8
    valid_size = 0.2

    torch.cuda.set_device(3)
    device = torch.device("cuda:3")
    print("now",torch.cuda.current_device())
    print(torch.cuda.device_count())
    base_path = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(base_path, "weights", "weight.pth")
    left_imglist, right_imglist, none_imglist = load_imglist()
    train_data_list = left_imglist + right_imglist + none_imglist
    # left_pivot = int(len(left_imglist)*train_size)
    # right_pivot = int(len(right_imglist)*train_size)
    # none_pivot = int(len(none_imglist)*train_size)
    # print(len(none_imglist))
    # print(len(left_imglist))
    # print(len(right_imglist))
    # train_data_list = left_imglist[:left_pivot] + right_imglist[:right_pivot] + none_imglist[:none_pivot]
    # valid_data_list = left_imglist[left_pivot:] + right_imglist[right_pivot:] + none_imglist[none_pivot:] 
    # train_data_list = train_data_list[:-10]
    # data_lists = left_imglist + right_imglist + none_imglist

    # train_label_pos = []
    # train_label_pos.append(len(left_imglist[:left_pivot]))
    # train_label_pos.append(len(left_imglist[:left_pivot])+ len(right_imglist[:right_pivot]))

    # valid_label_pos = []
    # valid_label_pos.append(len(left_imglist[left_pivot:]))
    # valid_label_pos.append(len(left_imglist[left_pivot:])+ len(right_imglist[right_pivot:]))

    label_pos = []
    #train_data_list = train_data_list[0:1]
    label_pos.append(len(left_imglist))
    label_pos.append(len(left_imglist) + len(right_imglist))
    train_loader, valid_loader = make_train_dataloader(train_data_list, label_pos)
    # train_loader = make_train_dataloader(train_data_list, train_label_pos)
    # valid_loader = make_train_dataloader(valid_data_list, valid_label_pos)

    print(len(train_loader.dataset)+len(valid_loader.dataset))
    # Classification criterion


    # Define network
    model = ALSTM(224,[49]*1,1,3)
    model = torch.nn.DataParallel(model, device_ids=[3,4,5,6]).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)



    # train
    train_loss_list = list()
    valid_loss_list = list()
    train_accuracy_list = list()
    valid_accuracy_list = list()
    best = 100
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs}')
        print('-' * len(f'Epoch: {epoch+1}/{epochs}'))
        train_loss, valid_loss = 0.0, 0.0
        train_correct, valid_correct = 0, 0
        train_accuracy, valid_accuracy = 0.0, 0.0

        model.train()
        for data, target in tqdm(train_loader, desc="Training"):
            
            if data.size(0) == 1:
                continue

            data, target = data.to(device), target.to(device)
            # Reset LSTM hidden state
            #model.module.lstm.reset_hidden_state()
            data = Variable(data,requires_grad=True )
            # forward + backward + optimize
            data = data.cuda()
            
            output  = model(data)
            _, preds = torch.max(output.data, 1)
            #print(output, target)

            loss = criterion(output, target)
            optimizer.zero_grad()   # zero the parameter gradients
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_correct += torch.sum(preds == target.data)
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
        train_accuracy = float(train_correct) / len(train_loader.dataset)
        train_accuracy_list.append((train_accuracy))


    model.eval()
    with torch.no_grad():
        for data, target in tqdm(valid_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            _, preds = torch.max(output.data, 1)

            valid_loss += loss.item() * data.size(0)
            valid_correct += torch.sum(preds == target.data)
        valid_loss /= len(valid_loader.dataset)
        valid_loss_list.append(valid_loss)
        valid_accuracy = float(valid_correct) / len(valid_loader.dataset)
        valid_accuracy_list.append((valid_accuracy))
    # print loss and accuracy in one epoch
    print(f'Training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}')
    print(f'Training accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}')

    # record best weight so far
    if valid_loss < best :
        best = valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    # save the best weight
    torch.save(best_model_wts, weight_path)

    # plot the loss curve for training and validation
    print("\nFinished Training")
    pd.DataFrame({
        "train-loss": train_loss_list,
        "valid-loss": valid_loss_list
    }).plot()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(1,epoch+1)
    plt.xlabel("Epoch"),plt.ylabel("Loss")
    plt.savefig(os.path.join(base_path, "result", "Loss_curve"))

    # plot the accuracy curve for training and validation
    pd.DataFrame({
        "train-accuracy": train_accuracy_list,
        "valid-accuracy": valid_accuracy_list
    }).plot()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(1,epoch+1)
    plt.xlabel("Epoch"),plt.ylabel("Accuracy")
    plt.savefig(os.path.join(base_path, "result", "Training_accuracy"))
    '''
    model = ALSTM(224,[49]*1,1,3)    
        
    model = model.cuda()
    for para in model.conv.parameters():
        para.requires_grad = False
    learning_rate = 1e-3  

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    
    accumulation_steps = int(14448/4)
    
    for t in range(10): # 
        
        train_loss=[]
        train_acc = 0
        #model.zero_grad()   
        train_mean = 0
        loader = ite.dataiter()  
        t_start = time.time()                               # Reset gradients tensors
        for j in range(accumulation_steps):
            print ("epoch: ",t)
            x_train,y_train = loader.next()
            x_train = Variable(x_train,requires_grad=True )
            y_train = Variable(y_train,requires_grad=True )
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            optimizer.zero_grad()
            
            y_pred,l = model(x_train)                    # Forward pass            
            my_loss,my_acc = my_cross_loss(l,y_train,y_pred,1)
            my_loss = my_loss / accumulation_steps                # Normalize our loss (if averaged)            
            with open("./loss.txt",'a+') as f:
                f.writelines(str(my_loss.item()))
                f.writelines('\n')
            
            my_loss.backward()        
            optimizer.step()                            # Now we can do an optimizer step           
            train_acc = train_acc + my_acc
            train_mean = train_mean + my_loss.item()
            print (j,' loss,acc are ',my_loss.item(),my_acc)
        print ('mean_loss',train_mean)
        t_end = time.time()
        delta = t_end-t_start
        print ("training time ",delta)
        with open("./train_loss.txt",'a+') as f:

                f.writelines(str(train_mean))
                f.writelines('\n')
           
    
    
    
    
        
        train_acc =  train_acc / accumulation_steps
        
        print ('training accuracy is ',train_acc)    
        with open("./train_acc.txt",'a+') as f:
            f.writelines(str(train_acc))
            f.writelines('\n')
        
        
        torch.save(model, './model_cfair10_2.pth')
        '''    # save model
    
    





 
if __name__ == '__main__':    
    train_batch()
    #train_seg()
    model = ALSTM(512,[49]*1,1,5)
    print (model)
    