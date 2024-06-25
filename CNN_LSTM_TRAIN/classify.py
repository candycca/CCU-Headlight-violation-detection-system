
import os
import cv2
import math
import random
import numpy as np
import datetime as dt
from collections import deque
import pandas as pd
from math import isnan
from PIL import Image


def add_img(start, end, temp, dir_path, videoname, imglist, carID, dirct):
    rename = False
    image_name = temp
    list = []
    if end - start + 1 < 30:
        return
    else:
        for i in range(start, start+30):
            name = image_name+ str(i) +'.jpg'
            path = os.path.join(dir_path, videoname, name)
            if not os.path.exists(path):
                labels = ['l', 'r', 's']
                for label in labels:
                    image_name = label + '_car' + str(carID) + "_"
                    name = image_name + str(i) +'.jpg'
                    path = os.path.join(dir_path, videoname, name)
                    if os.path.exists(path):
                        break
            
            img = Image.open(path)
            '''
            p = os.path.join("/home/ai113/code2/test", dirct,videoname)
            if not os.path.exists(p):
                
                os.makedirs(p)
            save_path = os.path.join(p,name)
            cv2.imwrite(save_path, img)
            '''
            list.append(img)
        imglist.append(list)


#pd.set_option('display.max_columns', None)
#显示所有行
#pd.set_option('display.max_rows', None)
def load_imglist():
    IMAGE_HEIGHT,IMAGE_WIDTH = 224,224
    SEQUENCE_LENGTH = 30

    CLASSES_LIST=["left", "right", "none"]
    file_path = "/home/ai113/code2/CNN_LSTM/LSTM_CNN_total.csv"
    dir_path = "/home/ai113/code2/test2VideoFolder/left-2_result/carimg"
    df_csv = pd.read_csv(file_path)
    print("loading_data...")
    #print(df_csv)
    left_imglist = []
    right_imglist = []
    none_imglist = []
    cols = df_csv.shape[1]
    rows = df_csv.shape[0]
    df_csv = df_csv.fillna(-1)
    for row in range(rows):
        a = df_csv.iloc[row].tolist()
        videoname = a[0]
        if videoname[0] == 'r' or videoname[0] == 'l':
            dir_path = "/home/ai113/code2/test2VideoFolder/"+ videoname +"_result/carimg"
            continue
        label = a[3]
        if not(label == "L" or label == "R" or label == 'S'):
            continue
        
        new_list = []
        for i in a[1:3]:
            new_list.append(int(i))
        for i in a[4:8]:
            if i == 'n':
                new_list.append(-1)
            else:
                new_list.append(int(i))
        for i in a[8:]:
            if i == -1:
                new_list.append(i)
            else:
                new_list.append([int(i.split('-')[0]), int(i.split('-')[1])])
        carID, newID, start, end, brake_on, brake_off, none, left, right = new_list

        image_name = (label.lower()) + '_car' + str(carID) + "_"
        if none != -1 :
            add_img(none[0], none[1], image_name, dir_path, videoname, none_imglist, carID, "none")
            
                
        if left != -1:
            add_img(left[0], left[1], image_name, dir_path, videoname, left_imglist, carID, "left")
            #print(newID)
            
            '''
            for i in range((left[0]), (left[1])+1):
                name = image_name+ str(i) +'.jpg'
                path = os.path.join(dir_path, videoname, name)
                img = cv2.imread(path)
                left_imglist.append([img])
                
            '''      
        if right != -1:
            add_img(right[0], right[1], image_name, dir_path, videoname, right_imglist, carID, "right")
            #print(newID)
    print("ok!")
    return left_imglist, right_imglist, none_imglist
#left:0, right:1, none:2
        

if __name__ == '__main__':    
    load_imglist()
    