
import os
import cv2
import ipdb
import glob
import numpy as np
from random import shuffle
from tf_unet import unet, util, image_util

os.environ["CUDA_VISIBLE_DEVICES"]="2"
TEST_SAMPLES = 5

#preparing data loading
data_provider = image_util.ImageDataProvider("/con_data/heatmap_work/train_unet/**.tif")

output_path = "/con_data/heatmap_work/train_logs/"


def _process_data(data):
    # normalization
    data = np.clip(np.fabs(data), 0, 255)
    data -= np.amin(data)

    if np.amax(data) != 0:
        data /= np.amax(data)

    return data

#setup & training

net = unet.Unet(cost = "heatmap_loss", layers=4, features_root = 64, channels = 1, n_class=2)
trainer = unet.Trainer(net,optimizer="adam", batch_size=2)
path = trainer.train(data_provider, output_path, training_iters=32, epochs=10,dropout=0.45) # probability to keep units)

# param x_test: Data to predict on. Shape [n, nx, ny, channels]
test_path = "/con_data/heatmap_work/val_unet/"

test_images = glob.glob(test_path + "**.jpg")
shuffle(test_images)

test_images = test_images[0:5] #

num_test_images = len(test_images)
x_test = np.zeros((num_test_images,800,800,1))

for img_idx in range(num_test_images):
  img_data = cv2.imread(test_images[img_idx])[:,:,0]
  img_data = _process_data(img_data)
  x_test[img_idx,:,:,0] = img_data


prediction = net.predict("/con_data/heatmap_work/train_logs/model.ckpt", x_test)

# Save tmp result images.
for test_idx in range(TEST_SAMPLES):
    class_1_map = prediction[test_idx,:,:,0]*255
    class_2_map = prediction[test_idx,:,:,1]*255

    cv2.imwrite("/con_data/heatmap_work/test_images/test_image_" + str(test_idx) + "_c1.jpg",class_1_map)
    cv2.imwrite("/con_data/heatmap_work/test_images/test_image_" + str(test_idx) + "_c2.jpg",class_2_map)
    cv2.imwrite("/con_data/heatmap_work/test_images/test_image" + str(test_idx) + ".jpg",x_test[test_idx,:,:,:]*255)



