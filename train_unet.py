
import os
import cv2
import ipdb
import glob
import numpy as np
from random import shuffle
from tf_unet import unet, util, image_util

os.environ["CUDA_VISIBLE_DEVICES"]="0"
TEST_SAMPLES = 10
IMAGE_SHAPE = (600,600)

#preparing data loading
data_provider = image_util.ImageDataProvider("/con_data/heatmap_work/train/**.tif", image_shape=IMAGE_SHAPE)

output_path = "/con_data/heatmap_work/train_logs/"

def _process_data(data):
    # normalization
    #data = np.clip(np.fabs(data), 0, 255)
    #data -= np.amin(data)

    #if np.amax(data) != 0:
    #    data /= np.amax(data)

    return data/255.

#setup & training
net = unet.Unet(cost = "dice_coefficient", layers=4, features_root = 64,
                channels = 1, n_class=2, cost_kwargs = {"class_weights":[0.0001333,0.999]})
trainer = unet.Trainer(net, optimizer="adam", batch_size=4,
                       opt_kwargs=dict(learning_rate =  0.0001))
path = trainer.train(data_provider, output_path, training_iters=32, epochs=50, dropout=0.55) # probability to keep units)


# param x_test: Data to predict on. Shape [n, nx, ny, channels]
test_path = "/con_data/heatmap_work/val/"

test_images = glob.glob(test_path + "**.tif")

filtered_test_images = []
# filter out the mask images.
for _test_image in test_images:
    if "_mask" not in _test_image:
        filtered_test_images.append(_test_image)

shuffle(filtered_test_images)

filtered_test_images = filtered_test_images[0:TEST_SAMPLES]

num_test_images = len(filtered_test_images)
x_test = np.zeros((num_test_images,IMAGE_SHAPE[0],IMAGE_SHAPE[1],1))

for img_idx in range(num_test_images):
  img_data = cv2.imread(filtered_test_images[img_idx])

  img_data = cv2.resize(img_data,IMAGE_SHAPE)
  img_data = _process_data(img_data)
  x_test[img_idx,:,:,0] = img_data[:,:,0]


prediction = net.predict("/con_data/heatmap_work/train_logs/model.ckpt", x_test)

# Save tmp result images.
for test_idx in range(TEST_SAMPLES):
    class_1_map = prediction[test_idx,:,:,0]*255
    class_2_map = prediction[test_idx,:,:,1]*255

    cv2.imwrite("/con_data/heatmap_work/test_images/test_image_" + str(test_idx) + "_c1.jpg",class_1_map)
    cv2.imwrite("/con_data/heatmap_work/test_images/test_image_" + str(test_idx) + "_c2.jpg",class_2_map)
    cv2.imwrite("/con_data/heatmap_work/test_images/test_image" + str(test_idx) + ".jpg",x_test[test_idx,:,:,:]*255)



