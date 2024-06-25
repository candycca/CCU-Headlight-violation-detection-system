from collections import defaultdict
import torch
import cv2
import os
from ultralytics import YOLO
import numpy as np
import csv
torch.cuda.set_device(3)
#device = torch.device("cuda:3")
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#創目錄用得函式，不用管他
def create_folder(name, parent=None):
    if parent:
        path = os.path.join(parent, name)
    else:
        path = name

    if not os.path.exists(path):
        os.makedirs(path)
    return path

# input:
#1.input_folder_path:存放你想預測的影片的資料夾路徑，裡面不能有子資料夾！
#2.car_img_folder_path:存放車輛序列照片的資料夾之path
#3.box_info_folder_path:存放bounding_box資料的資料夾
# output:
#1.各video的{yolo預測結果}會以(video_name)_ouput.mp4存在code2資料夾下，我本來想把他們用程式統一到某個資料夾，但失敗，所以我現在把所有的output_video放在code2/video_output底下
#2.{每部影片每個id出現的幀數及其在該幀數的bounding box之中心座標以及box長寬}以txt檔存在box_info_folder_path下
#3.{每部影片所有車輛序列照片}存在car_img_folder_path資料夾底下，例如你要找v25中id為215的車的照片，可以在{car_img_folder_path}/v25/car215中找到


def Yolo_Car_Predict(model, filename,video_path, box_info_folder_path, output_folder, cut_with_outside):
    
    #output_folder
    info_folder = create_folder(box_info_folder_path) #存{每個id出現的幀數及其在該幀數的bounding box之中心座標以及box長寬}的資料夾
    img_list = [[]for _  in range(1000)] #車輛序列的列表
    # 遍歷資料夹中的所有文件
    #合成video_path
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error reading video file: {video_path}")
        return

    #output_video的width, height, fps設定
    width, height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    #合成output_video_path路徑
    if not os.path.exists(output_folder + "/videoOutput"):
        os.makedirs(output_folder + "/videoOutput")
    #output_video_path = output_folder + "/videoOutput/"+filename[:-4]+"_output.mp4"
    
    #video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    #紀錄追蹤資訊
    track_info = defaultdict(lambda: {"frames": [], "bboxes": []})
    frame_num = 0
    #讀幀
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_num += 1
            #yoloV8啟動
            results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes = [2,7])
            #要先確認影片中是否有偵測到物件
            if results[0].boxes.id is not None:
                #boxes為這一幀所有bounding box資訊（中心座標以及w,h)得集合
                #track_ids為這一幀所有id得集合
                
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                #將預測結果寫入影片（就是那些框框）
                annotated_frame = results[0].plot()
                cv2.putText(annotated_frame, str(frame_num), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
                
                #video_writer.write(annotated_frame)
                
                #將boxes和track_ids中的資料放入track_info
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track_info[track_id]["frames"].append(frame_num)
                    track_info[track_id]["bboxes"].append((x, y, w, h))
                    #計算bounding box左上角及右下角座標以供opencv截圖
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x1 + w)
                    y2 = int(y1 + h)
                    
                    if cut_with_outside == 1:
                        green_color = (0,255,0)
                        roi = np.ones((frame.shape[0],frame.shape[1],frame.shape[2]), np.uint8) * 255
                        roi[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                        if y1 != 0:
                            roi[y1-1, x1:x2] =  green_color
                        if x1 != 0:
                            roi[y1:y2, x1-1] =  green_color
                        if y2 != frame.shape[0] - 1 and y2 != frame.shape[0]:
                            roi[y2+1, x1:x2] =  green_color
                        if x2 != frame.shape[1] - 1 and x2 != frame.shape[1]:
                            roi[y1:y2, x2+1] =  green_color

                    elif cut_with_outside == 2 :
                        if x1 - 10 > 0:
                            x1 -= 10
                        else:
                            x1 = 0
                        if frame.shape[1] - x2 - 10 > 0:
                            x2 += 10
                        else:
                            x2 = frame.shape[1] - 1
                        
                        roi = frame[y1:y2, x1:x2]
                        roi = cv2.resize(roi, (224, 224))
                    else:
                        roi = frame[y1:y2, x1:x2]
                    #合成img輸出path
                    img_list[track_id].append(roi)
            else:
                continue
        else:
            break
    model.predictor.trackers[0].reset()
    cap.release()
    #video_writer.release()
    cv2.destroyAllWindows()
    #寫txt檔
    output_file = os.path.join(info_folder, f"{filename[:-4]}_track_info.txt")
    output_csvfile = os.path.join(info_folder, f"{filename[:-4]}_track_info.csv")
    with open(output_file, "w") as f:
        for track_id, info in track_info.items():
            frames = info["frames"]
            bboxes = info["bboxes"]
            f.write(f"Track ID: {track_id}\n")
            f.write("Bounding Boxes:\n")
            for bbox in bboxes:
                x, y, w, h = bbox
                f.write(f"    Frame: {frames[bboxes.index(bbox)]}, Bounding Box: ({x}, {y}, {w}, {h})\n")

    with open(output_csvfile, 'w', newline='') as csvoutput:
        writer = csv.writer(csvoutput)
        for track_id, info in track_info.items():
            frames = info["frames"]
            bboxes = info["bboxes"]
            for bbox in bboxes:
                x, y, w, h = bbox
                frameNum= frames[bboxes.index(bbox)]
                csvoutput.write(f'{track_id},{frameNum},{x:.17f},{y:.17f},{w},{h}\n')
                
    return img_list



