import os
import cv2

# Import and organizing chest-xray-pneumonia-dataset
os.system("kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
os.system("unzip chest-xray-pneumonia.zip")
os.system("mv chest_xray chest-xray-pneumonia-dataset")
os.system("rm -rf chest-xray-pneumonia.zip")

