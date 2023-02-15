import cv2
import torch
from PIL import Image
from pathlib import Path
# Model
'''
     This script was written for yolo model testing on a single image
    'get_yolov5' will create a the custom model by our pre-trained custom weight.
     then a image will be given in Image.open('folder location')
     results.display will take multiple argments, if arguments are given as true
     results will be shown for that parameters only
'''
def get_yolov5():
    model = torch.hub.load("F:/foruse_only_muzzle_detector_yolo/aps_muzzle_detector", 'custom', 
            path="F:/foruse_only_muzzle_detector_yolo/aps_muzzle_detector/runs/weights/YOLO_updated_jan_2023.pt", source='local')
    model.conf=0.7
    return model

# Images
im1 = cv2.imread('C:/Users/IT BD/Downloads/Book.JPEG')  
im1=cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)

# Inference
final_model = get_yolov5()
results = final_model(im1, size=640)  # includes NMS
results.display(pprint=True, show=True, save=True, crop=False, render=True, labels=True) #save_dir=Path('E:/v2_data/2_not_detected'))
print(results.xyxy[0]) # im predictions (tensor)
print(results.pandas().xyxy[0])
