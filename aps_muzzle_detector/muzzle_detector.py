import torch
from PIL import Image
import os

root = os.path.dirname(os.path.abspath(__file__))
yolo_relative_weight_path =  "runs/weights/yolo.pt"
yolo_weight_path = os.path.join(root, yolo_relative_weight_path)
model_confidence = 0.7

# Sending image to this function for muzzle detection
def crop_muzzle(image, yolo_folder_path=root, yolo_weigth_path=yolo_weight_path, model_confidence=model_confidence):
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

    '''
    # creating the custom model function
    def get_yolov5():
        model = torch.hub.load(yolo_folder_path, 'custom', path=yolo_weigth_path, source='local')
        model.conf = model_confidence
        return model 
    final_model = get_yolov5()
    results = final_model(image, size=640)

    # croping the detected muzzle part
    crops = results.display(pprint=True, show=False, save=False, crop=True, render=True, labels=True, 
                        save_dir=False)

    num_muzzle = len(crops) # counting how many muzzle got from an image
    muzzle_dictionary = {"num_crops": num_muzzle,  "image": None} 
    if muzzle_dictionary["num_crops"] == 1:
        image_array = crops[0]["im"]
        img_read = Image.fromarray(image_array) # converting numpay array to a PIL image 
        muzzle_dictionary["image"] = img_read
    return muzzle_dictionary # return the dictionary with the keys "num_crops" and "image" and their values
