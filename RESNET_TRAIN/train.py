import torch
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from model import ResNet
from model import ResidualBlock
import gc
device = torch.device('cuda:3')
#train data:
from dataLoad import make_train_dataloader
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator



# training parameters
epochs = 20
learning_rate = 0.01

# data path and weight path
base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "data", "train")
weight_path = os.path.join(base_path, "weights", "weight.pth")

# make dataloader for train data
train_loader, valid_loader = make_train_dataloader(train_data_path)

# set cnn model
model = ResNet(ResidualBlock, [3,4,6,3])
model = model.to(device)

# set optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

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
        data, target = data.to(device), target.to(device)

        # forward + backward + optimize
        output  = model(data)
        _, preds = torch.max(output.data, 1)
        #print(output, target)
        #log_output =  nn.LogSoftmax(dim=1)(output)
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