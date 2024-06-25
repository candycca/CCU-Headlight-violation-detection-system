import cv2 as cv
import pandas as pd
import os
import glob
import numpy as np
import sys

from ResNet.test import predict
def draw_img(output_path,points,id):
    img = np.zeros((540,960,3), np.uint8)
    points = np.array(points)
    points = points.astype(np.int32).reshape((-1, 1, 2))
    output_path =  output_path + "/" + id + ".jpg"
    cv.polylines(img, [points], isClosed=False, color=(255, 255, 255), thickness=1)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(output_path, gray_img)

def draw(input_path,output_path):
    df = pd.read_csv(input_path, header = None)
    num = df.iloc[-1,0]
    for i in range(1,num+1):
        result = df[df.iloc[:, 0] == i]
        points = []
        if (result.shape[0] != 0) :
            x1 = int(result.iloc[0,2])
            y1 = int(result.iloc[0,3])
            x2 = int(result.iloc[-1,2])
            y2 = int(result.iloc[-1,3])
            #起點到終點距離
            dis = (y2 - y1)**2 + (x2 - x1)**2
            #車輛出現長度大於一秒and行駛方向正確and移動長度大於10000（移動待確認）
            if result.shape[0] > 35 and (y2 - y1) < 0 and dis > 10000:
                for j in range(result.shape[0]):
                    points.append([int(result.iloc[j,2]),int(result.iloc[j,3])])
                #print(points,result.shape[0])
                print(i)
                num = str(i)
                draw_img(output_path,points,num)
            else:
                continue


def find_turn(name,input_folder,turn_info_folder,type = "turn"):

    turnCar = []
    right_turn_car = []
    left_turn_car = []
    striaght_car = []
    if not os.path.exists(turn_info_folder):
        os.makedirs(turn_info_folder)
    ResNet_result_folder = os.path.join(turn_info_folder, 'ResNet_result')#存resNet
    track_img_folder = os.path.join(turn_info_folder,'track_img', name, '_track_img')#存每部影片軌跡圖的資量夾
    if not os.path.exists(ResNet_result_folder):
        os.makedirs(ResNet_result_folder)
    if not os.path.exists(track_img_folder):
        os.makedirs(track_img_folder)
    draw(input_folder,track_img_folder)
    path = os.listdir(track_img_folder)
    if (len(path) == 0):
        return turnCar
    else:
        turnCar = predict(track_img_folder, ResNet_result_folder, name)
        #turnCar = [left_car,right_car,straight_car]
        return turnCar






