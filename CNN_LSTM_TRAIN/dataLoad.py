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
from torch.utils.data import DataLoader, Dataset

# You can modify batch size here
train_batch_size = 16
test_batch_size = 16
num_workers = 0
train_size_rate = 0.8   # Split dataset into train and validation 8:2
image_shape = (3, 224, 224)


'''
class Dataset(Dataset):
    def __init__(self, data_list, input_shape, label_pos = [], training=True):
        # 初始化参数
        self.data_list = data_list
        self.input_shape = input_shape
        self.label_pos = label_pos  #LRN(label 012) label_num[0]:R開始的位置 label_num[1]:N開始的位置 
        self.training = training

        # 数据转换操作
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 获取图像路径
        image_list = self.data_list[idx]
        
        # 应用转换
        if(not torch.is_tensor(image_list[0])):
            for frame in range(len(image_list)):
                image_list[frame] = self.transform(image_list[frame])
        
        # 生成标签
        if self.training:
            if(idx < self.label_pos[0]):
                label = 0
            elif(idx < self.label_pos[1]):
                label = 1
            else:
                label = 2
            return torch.stack(image_list), torch.tensor(label)
        else:
            
            return torch.stack(image_list)
'''
class Dataset(Dataset):
    def __init__(self, data_list, input_shape, label_pos, id, window, step = 1, training=True):
        # 初始化参数
        self.data_list = data_list
        self.input_shape = input_shape
        self.label_pos = label_pos  #LRN(label 012) label_num[0]:R開始的位置 label_num[1]:N開始的位置 
        self.training = training
        self.id = id
        self.window = window
        self.step = step
        # 数据转换操作
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        print(len(self.data_list) - self.window)
        return max(0,len(self.data_list) - self.window)
        

    def __getitem__(self, idx):
        # 获取图像路径
        windows_s = idx
        windows_t = idx + self.window
        image_list = self.data_list[windows_s:windows_t-1]
        # 应用转换
        if(not torch.is_tensor(image_list[0])):
            for frame in range(len(image_list)):
                image_list[frame] = self.transform(image_list[frame])
        
        # 生成标签
        if self.training:
            if(self.id < self.label_pos[0]):
                label = 0
            elif(self.id < self.label_pos[1]):
                label = 1
            else:
                label = 2
            return torch.stack(image_list), torch.tensor(label)
        else:
            return torch.stack(image_list)
    
def make_train_dataloader(data_list, label_pos, window = 30, step = 1):
    id = 0
    for data in data_list:
        if id == 0:
            dataset = Dataset(
                data_list = data, 
                input_shape = image_shape, 
                label_pos = label_pos, 
                id = id,
                window= window,
                step = step,
                training=True
                )
        else:
            dataset += Dataset(
                data_list = data, 
                input_shape = image_shape, 
                label_pos = label_pos, 
                id = id,
                window= window,
                step = step,
                training=True
                )
        id += 1
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    

    return data_loader


def make_test_dataloader(data, window = 30, step = 1):
    test_dataset = Dataset(
                data_list = data, 
                input_shape = image_shape, 
                label_pos = 0, 
                id = 0,
                window= window,
                step = step,
                training=False
    )
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    return test_loader





# def make_actual_dataloader(data_path):

#     testData = torch.stack(laod_actual_data(data_path,transform=test_transforms)) # For converting list to tensor
#     # testData = transform(testData)
#     test_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=4)
#     return test_loader

# def laod_actual_data(data_path, transform =None):
#     images = []
#     path_list = os.listdir(data_path)
#     path_list.sort(key=lambda x:int(x.split('.')[0]))

#     for file_name in path_list:
#         img_path = os.path.join(data_path, file_name)
#         img = Image.open(img_path).convert('RGB')
#         if transform:
#             img = transform(img)
#         images.append(img)
#     return images

