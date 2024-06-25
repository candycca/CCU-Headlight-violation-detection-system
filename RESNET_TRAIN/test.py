import torch
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from ResNet.model import ResNet
from ResNet.model import ResidualBlock
import gc

#train data:
from ResNet.dataLoad import make_train_dataloader
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from ResNet.dataLoad import make_test_dataloader
from ResNet.dataLoad import make_actual_dataloader

class_names = ['left', 'right', 'straight']

device = torch.device('cuda:0')

def predict_test_data(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend([class_names[p] for p in predicted.cpu().numpy()])

    return predictions

def predict(test_data_path,output_path,name):
    weight_path = "/home/ai113/code2/ResNet/weights/weight.pth"
    # load model and use weights we saved before
    model = ResNet(ResidualBlock, [3,4,6,3])
    model.load_state_dict(torch.load(weight_path,  map_location='cuda:0'))
    model = model.to(device)

    # make dataloader for test data
    test_loader = make_actual_dataloader(test_data_path)
    predictions = predict_test_data(model, test_loader)
    path_list = os.listdir(test_data_path)
    path_list.sort(key=lambda x:int(x.split('.')[0]))
    dfDict = { 
        'file': path_list,
        'species': predictions
    }
    left_car = []
    right_car = []
    straight_car = []
    for key,value in zip(path_list, predictions):
        if value == 'left':
            left_car.append(int(key.split('.')[0]))
        elif value == 'right':
            right_car.append(int(key.split('.')[0]))
        elif value == 'straight':
            straight_car.append(int(key.split('.')[0]))
    
    df = pd.DataFrame(dfDict)
    csv_file_path = os.path.join(output_path, name+"_predictions.csv")
    df.to_csv(csv_file_path, index=False)

    print(f"Predictions saved to {csv_file_path}")
    return [left_car,right_car,straight_car]


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(base_path, "data", "test")
    weight_path = os.path.join(base_path, "weights", "weight.pth")

    # load model and use weights we saved before
    model = ResNet(ResidualBlock, [3,4,6,3])
    model.load_state_dict(torch.load(weight_path))
    model = model.to(device)

    # make dataloader for test data
    test_loader = make_test_dataloader(test_data_path)

    predictions = predict_test_data(model, test_loader)

    dfDict = {
        'file': os.listdir(test_data_path),
        'species': predictions
    }

    df = pd.DataFrame(dfDict)

    csv_file_path = os.path.join(base_path, "predictions.csv")
    df.to_csv(csv_file_path, index=False)

    print(f"Predictions saved to {csv_file_path}")

if __name__ == "__main__":
    main()