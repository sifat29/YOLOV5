from logging import raiseExceptions
import cv2
import torch
from PIL import Image
import os

'''
    'crop_muzzle' will detect muzzle from an Image and return the dictionary with "num_crops" and "image".
    'get_yolov5' will create a the custom model by our pre-trained custom weight.
    'var: crops' After detecting the muzzle form image this will crop the detected part.

    Args:
    ----
    image: An image (recommended .jpg format)
        image will take an image from a directory. 
    yolo_folder_path: str
        Will take the root path of the yolo folder
    yolo_weight_path: str
        Will take the pretrained custom weight file path
    model_confidence: float or double
        Will take the confidence as float value from range: 0 to 1
     at the end it will return the crop image in the saved folder
 '''

clean_path = r"E:/Day_2/R_D_12"
save_path = r"E:/day2_croped"
image_not_detected = r"E:/v2_data/2_not_detected"


yolo_folder_path = "F:/foruse_only_muzzle_detector_yolo/aps_muzzle_detector"
yolo_weigth_path = "F:/foruse_only_muzzle_detector_yolo/aps_muzzle_detector/runs/weights/YOLO_updated_jan_2023.pt"

model_confidence = 0.1


def crop_muzzle(image, yolo_folder_path, yolo_weigth_path, model_confidence, destination):
    def get_yolov5():
        model = torch.hub.load(yolo_folder_path, 'custom', path=yolo_weigth_path, source='local')
        model.conf = model_confidence
        return model
    
    final_model = get_yolov5()
    results = final_model(image, size=640)
    crops = results.display(pprint=True, show=False, save=False, crop=True, render=True, labels=True, 
                        save_dir=False)

    
    num_muzzle = len(crops)
    if num_muzzle == 1:
        image_array = crops[0]["im"]
        image1 = cv2.imwrite(destination, image_array)
        return image1

    elif num_muzzle == 0:
        return None
    else:
        print("Multiple Muzzle Detected")

    
path_dirs = list(os.walk(clean_path))[1:]

for dirs in path_dirs:
   
    dir = dirs[0]
    files = dirs[2]
    savedir = os.path.join(save_path, os.path.basename(dir))

    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    for name in files:
        source = os.path.join(clean_path, os.path.basename(dir), name)
        image = cv2.imread(source)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(image)
        destination = os.path.join(savedir, name)
        crop_img = crop_muzzle(image, yolo_folder_path, yolo_weigth_path, model_confidence, destination)
        if crop_img == None:
            image.save(os.path.join(image_not_detected, name))
            
         