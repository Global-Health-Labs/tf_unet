"""
Script to resize images given in a folder. 

SK
"""

import os
import cv2
import glob 
import numpy as np
from PIL import Image
from random import shuffle

folder_path = "/con_data/heatmap_work/train_unet/" 
RESIZE_TO = (800,800)
NUM_CLASSES = 2

images = glob.glob(folder_path + "*_mask.jpg") 
shuffle(images)

for _image in images:
  img_data = cv2.imread(_image[:-9] + ".jpg")
  img_data_tif = Image.fromarray(img_data)
  img_data_tif.save(_image[:-9] + ".tif") 

  mask_data = cv2.imread(_image)
  mask_data_tif = np.zeros((800,800,NUM_CLASSES),dtype = np.uint8)
  mask_data_tif[:,:,0] = mask_data[:,:,0]
  mask_data_tif[:,:,1] = 255 - mask_data[:,:,0]

  mask_data_save_tif = Image.fromarray(mask_data_tif)
  
  #mask_data_save_tif = Image.fromarray(mask_data_tif) 
  mask_data_save_tif.save(_image[:-3] + "tif")


 
  #img_data_resized = cv2.resize(img_data,RESIZE_TO) 
  #mask_data_resized = cv2.resize(mask_data,RESIZE_TO)

  #cv2.imwrite(_image[:-9] + ".jpg",img_data_resized)
  #cv2.imwrite(_image,mask_data*255)






