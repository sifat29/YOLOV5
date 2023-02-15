import os
from random import choice
import shutil

crsPath = "E:/yolo_annotation/train_data2/images/train"
labelPath = "E:/yolo_annotation/train_data2/labels/train"
for (dirname, dirs, files) in os.walk(crsPath):
for txtfile in (os.path.join(crsPath, "*.txt")):
    shutil.copy(txtfile, labelPath)
# print(xmls)
# source=os.path.join(crsPath+str(xmls))
# destination=os.path.join(labelPath+str(xmls))
# shutil.move(source, destination)
# print(xmlsfile)
            # shutil.move(os.path.join(crsPath, xmls), os.path.join(labelPath, xmlsfile))