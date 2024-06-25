from collections import defaultdict
#import tensorflow as tf
import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import csv

print("jjj")
#model = YOLO('/home/ai113/runs/detect/train9/weights/best.pt')
#results = model.predict(source='code2/carImg/v1_video/car1',  save = True, conf = 0.4)
#result = model.predict(source='code2/carImg/v1_video/car1', tracker = 'byte', save=True)
# 读取图像
'''
input_folder = "/home/ai113/code2/carImg/20230416_123208_2709_A_video/car11"
os.chdir(input_folder)
num_list=[]
model = YOLO('/home/ai113/runs/detect/train14/weights/best.pt')
for filename in os.listdir(input_folder):
    results = model.predict(source=filename)
    num_list.append(len(results[0].boxes))
print(num_list)
plt.plot(num_list)
plt.title(f"light Information")
plt.xlabel('Index')
plt.ylabel('light')
os.chdir('/home/ai113')
plt.savefig(f'light_info.png')
plt.cla()
'''

def writeCSV(writer, Nums):
    row0 = []
    row1 = []
    for num in Nums:
        if num > 0:
            if len(row1) != 0:
                writer.writerow(row1)
                
                #print(row1)
                row1 = []
            row0.append(num)
        else:
            if len(row0) != 0:
                writer.writerow(row0)
                #print(row0)
                row0 = []
            row1.append(num)

    if(len(row0)!= 0):
        writer.writerow(row0)
    if(len(row1)!= 0):
        writer.writerow(row1)

        
        









def turn_light_Predict(model, name, carID, img_list, turn_light_info_dir):
    
    #print(path_list)
    num_list=[]
    # for filename in path_list:
    #     '''
        
    #     projects = '/home/ai113/code2/tryIMG'
    #     if not os.path.exists(projects):
    #         os.makedirs(projecst)
    #     subproject = projects + '/' + name.split('.')[0]
    #     if not os.path.exists(subproject):
    #         os.makedirs(subproject)
    # '''
    #     results = model.predict(source=filename, save = True)
    #     if len(results[0].boxes) == 0:
    #         num_list.append(len(results[0].boxes))
    #     else:
    #         num_list.append(1)
    for i in img_list:
        results = model.predict(i,save=True, project = '/home/ai113/code2/onlyOneVideoFolder/video_test_result/CAresult')
        if len(results[0].boxes) == 0:
            num_list.append(len(results[0].boxes))
        else:
            num_list.append(1)
    light = is_light(num_list)
    #print(carID, light)
    plt.plot(num_list)
    plt.title(f"light Information")
    plt.xlabel('Index')
    plt.ylabel('light')
    
    if not os.path.exists(turn_light_info_dir):
        os.makedirs(turn_light_info_dir)
    os.chdir(turn_light_info_dir)

    tmp_path = name.split('.')[0]
    output_dir = tmp_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_output = tmp_path + "/car" + str(carID) + ".csv"
    with open(csv_output, mode='w', newline='') as output_file:
        writer = csv.writer(output_file)
        writeCSV(writer, num_list)
        writer.writerow(['light',light])
    
    output_path = tmp_path + "/car" + str(carID) + ".jpg"
    plt.savefig(output_path)
    plt.cla()      
    
    


#判斷打燈(0:沒有打燈 1:有打燈)
def is_light(num_list):
    upper = 0
    lower = 0
    wave_num = 0
    frame = len(num_list)
    flag = 0
    flag2 = 0
    flag3 = 0
    #改成除以第一個出現1的地方(前面車直行不計)
    for i in num_list:
        if i == 1:
            upper += 1
            flag = 1
        elif i == 0 and flag == 1:
            lower += 1
    for i in num_list:
        if i == 1 and flag2 == 0:
            flag2 = 1
        elif i == 1 and flag2 == 1 and flag3 == 0:
            wave_num += 1
            flag3 = 1
        elif i == 0:
            flag2 = 0
            flag3 = 0
    #錯誤修正(如果是一直線會除以0，所以要避免)
    if upper == 0 and lower == 0:
        lower = 1
    print(upper,lower, wave_num)
    print(upper, lower, upper / (upper + lower), wave_num/frame)
    #先設0.5, 0.04(每秒一個波)之後再改
    if upper / (upper + lower) < 0.5 and wave_num/frame < 0.04:
        light = 0 
    else:
        light = 1

    return [light, upper / (upper + lower), wave_num/frame]




def check(csv_path):
    with open(csv_path, newline = '') as csvfile:
        csv_reader = csv.reader(csvfile)
        islight = 0
        for row in csv_reader:
            count = 0
            for char in row:
                print(char, end = ' ')
                if char == '1':
                    count += 1
            if count >= 2:
                islight+=1
            print()

        if islight >= 2:
            print("islight")
        else:
            print("no light")










    










