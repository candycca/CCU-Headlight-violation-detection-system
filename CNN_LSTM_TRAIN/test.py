import torch
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import gc

from CNN_LSTM.CNNLSTM import *
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from CNN_LSTM.ccctry import make_test_dataloader

class_names = ['left', 'right', 'none']



# testing parameters

image_shape = (3, 224, 224)
num_classes = 3
latent_dim =  512
lstm_layers =  3
hidden_dim =  1024
bidirectional =  True
attention = True

device = torch.device('cuda:2')

def predict_test_data(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for image_list in tqdm(test_loader, desc="Predicting"):
            image_list = image_list.to(device)
            model.lstm.reset_hidden_state()
            outputs = model(image_list)
            _, predicted = torch.max(outputs, 1)
            predictions.extend([class_names[p] for p in predicted.cpu().numpy()])

    return predictions

#car_list車輛序列、
def CNN_LSTM_predict(car_list,output_path,carID):
    print('kkk')
    weight_path = "/home/ai113/code2/CNN_LSTM/weights/weight.pth"
    # load model and use weights we saved before
    model = CNNLSTM(
        num_classes = num_classes,
        latent_dim = latent_dim,
        lstm_layers = lstm_layers,
        hidden_dim = hidden_dim,
        bidirectional = bidirectional,
        attention = attention,
    )
    model.load_state_dict(torch.load(weight_path,  map_location='cuda:0'),False)
    model = model.to(device)
    # make dataloader for test data
    
    test_loader = make_test_dataloader(car_list)
    predictions = predict_test_data(model, test_loader)
    left = 0
    right = 0
    none = 0
    for result in predictions:
        if result == 'left':
            left += 1
        if result == 'right':
            right += 1
        if result == 'none':
            none += 1
    length = len(predictions)
    dfDict = { 
        'ID': list(range(1, length + 1)),
        'prdict': predictions
    }
    df = pd.DataFrame(dfDict)
    left_df = pd.DataFrame({'ID': ['left'],'prdict': [left]})
    df = pd.concat([df, left_df], ignore_index=True, axis=0)
    right_df = pd.DataFrame({'ID': ['right'],'prdict': [right]})
    df = pd.concat([df, right_df], ignore_index=True, axis=0)
    none_df = pd.DataFrame({'ID': ['none'],'prdict': [none]})
    df = pd.concat([df, none_df], ignore_index=True, axis=0)
    csv_file_path = os.path.join(output_path, str(carID) + "_predictions.csv")
    df.to_csv(csv_file_path, index=False)

    print(f"Predictions saved to {csv_file_path}")
    return left, right, none



def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(base_path, "data", "test")
    weight_path = os.path.join(base_path, "weights", "weight.pth")

    # load model and use weights we saved before
    model = CNNLSTM(
        num_classes = num_classes,
        latent_dim = latent_dim,
        lstm_layers = lstm_layers,
        hidden_dim = hidden_dim,
        bidirectional = bidirectional,
        attention = attention,
    )
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