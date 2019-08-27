"""
Script to inverse the image channel to improve PE results. 
Second channel is the inverse of first channel. Third is same as first. 

-SK
"""

import os
import glob
import cv2

image_path = "/con_data/heatmap_work/val_unet_inv/"

images = glob.glob(image_path + "*.tif") 

filter_images = [] 

# filter out _mask images
for _image in images:
  if "_mask" not in _image:
    filter_images.append(_image)

for _image in filter_images: 
  img_data = cv2.imread(_image) 
  img_data[:,:,1] = 255 - img_data[:,:,1]

  cv2.imwrite(_image,img_data)

