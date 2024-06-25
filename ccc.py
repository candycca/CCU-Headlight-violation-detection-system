from  YoloTrack_try4 import Yolo_Car_Predict
from  findTurn2 import findTurn #old
from ultralytics import YOLO
from light_predict import turn_light_Predict #old #turn_light_Predict(light_model, filename, carID, img_list[carID], turn_light_info_dir)
import os
import cv2
from screenshot import get_the_resultIMG
import time
from ResNetTurn import find_turn #turn
import pandas as pd
import shutil
import argparse
from CNN_LSTM.test import CNN_LSTM_predict #light
import numpy as np
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--name', type = str, default="")
args = parser.parse_args()

#args_name = "/home/ai113/code2/onlyOneVideoFolder/video_test"
       

#製作背景全白的資料集 0：原本的方法/1: 補背景/2: resize和yolo bbox + 10個像素
cut_with_outside = 2


start_time = time.time()
input_folder = "/home/ai113/code2/" + args.name
#output_folder
output_folder = "/home/ai113/code2/" + args.name + "_result"
info_folder = output_folder + "/id-box-info"
turn_info_folder = output_folder + '/turn_info'
turn_light_info_dir = output_folder + '/light_info'
carimg_folder = output_folder + '/carimg'
result_folder = output_folder + '/result'


#model
#light_model = YOLO('/home/ai113/runs/detect/train2_CA/weights/best.pt')
car_model = YOLO('yolov8n.pt')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(carimg_folder):
    os.makedirs(carimg_folder)

if not os.path.exists(turn_light_info_dir):
    os.makedirs(turn_light_info_dir)


for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        #合成video_path
        video_path = os.path.join(input_folder, filename)

        #車輛偵測(Yolo track try4)
        img_list = Yolo_Car_Predict(car_model,filename, video_path, info_folder ,output_folder, cut_with_outside)

        #下面這一部分是轉彎偵測(ResNet)
        '''
        #detect turn
        turnCars = []
        #舊方法：turnCar = findTurn(info_folder,f"{filename[:-4]}_track_info.txt", turn_info_folder)
        input_path = os.path.join(info_folder,f"{filename[:-4]}_track_info.csv")
        #使用ResNet的新方法
        turnCars = find_turn(filename[:-4],input_path,turn_info_folder)              
        #detect light
        #先檢查左轉再檢查右轉
        #turncars = [left_car,right_car,straight_car]
        '''

        #下面這一部分是CNN-LSTM偵測
        '''
        n = 0
        for turnCar in turnCars:
            if n == 0:
                dir = 'l'
            elif n == 1:
                dir = 'r'
            elif n == 2:
                dir = 's'

            if len(turnCar) == 0:
                    print("no car")
                    
            else:
                for carID in turnCar:
                    #carID = turnCar[i]
                    print(carID)
                    

                    ###################
                    #額外寫的之後改函式，記得註解
                    #outputImage
                    # os.chdir(carimg_folder)
                    # if not os.path.exists(filename[:-4]):
                    #     os.makedirs(filename[:-4])
                    # os.chdir(filename[:-4])
                    # count = 0
                    # for i in img_list[carID]:
                    #     count += 1
                    #     carimg_path = carimg_folder + "/" + filename[:-4] + "/" + dir + "_car" + str(carID) +"_" + str(count) +".jpg"
                    #     cv2.imwrite(carimg_path,i)
                        

                    ########################
                    #path = turn_info_folder+"/turn_data/"+filename.split('.')[0]+'.csv'
                    #CNN-LSTM預測模型
                    
                    filename_path = os.path.join(turn_light_info_dir,filename[:-4])
                    if not os.path.exists(filename_path):
                        os.makedirs(filename_path)
                    CNN_LSTM_predict(img_list[carID],filename_path,carID) 
               
                    
            n = n + 1  
        '''      
                    #get_the_resultIMG(path, filename, total_frame)
        




        
                  
            

            
                


end_time = time.time()
duration_seconds = end_time - start_time

print("Time：", duration_seconds, "秒")  