import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from PIL import Image
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# You can modify batch size here
train_batch_size = 32
test_batch_size = 32
num_workers = 0
train_size_rate = 0.8   # Split dataset into train and validation 8:2

# Image transform for train and test
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Make dataset and dataloader(Don't need to adjust)
def make_train_dataloader(data_path):

    dataset = datasets.ImageFolder(root=data_path, transform=data_transforms)
    train_size = int(len(dataset) * train_size_rate)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader

def make_test_dataloader(data_path):

    testData = torch.stack(load_test_data(data_path,transform=test_transforms)) # For converting list to tensor
    # testData = transform(testData)
    test_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=4)
    return test_loader

def load_test_data(data_path, transform=None):
    images = []
    for file_name in os.listdir(data_path):
        img_path = os.path.join(data_path, file_name)
        img = Image.open(img_path).convert('RGB')
        if transform:
            img = transform(img)
        images.append(img)
    return images



def make_actual_dataloader(data_path):

    testData = torch.stack(laod_actual_data(data_path,transform=test_transforms)) # For converting list to tensor
    # testData = transform(testData)
    test_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=4)
    return test_loader

def laod_actual_data(data_path, transform =None):
    images = []
    path_list = os.listdir(data_path)
    path_list.sort(key=lambda x:int(x.split('.')[0]))

    for file_name in path_list:
        img_path = os.path.join(data_path, file_name)
        img = Image.open(img_path).convert('RGB')
        if transform:
            img = transform(img)
        images.append(img)
    return images

