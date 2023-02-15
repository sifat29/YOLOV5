import cv2
import numpy as np

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
# Get Aruco marker
img = cv2.resize(img, (0,0), None, 0.5, 0.5)
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
print("this the corners",corners)
# Aruco Perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)

# Pixel to cm ratio
pixel_cm_ratio = aruco_perimeter / 20
# Get Aruco marker
