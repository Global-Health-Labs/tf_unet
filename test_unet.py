"""
Script to test freshly trained Unet model. 

"""

import os 
import glob 
import ipdb
import cv2 
import numpy as np
import tensorflow as tf 
from random import shuffle

model_path = "/con_data/heatmap_work/train_logs/model.ckpt" 

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def restore(sess, model_path):
    """
    Restores a session from a checkpoint
    :param sess: current session instance
    :param model_path: path to file system checkpoint location
    """

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    logging.info("Model restored from file: %s" % model_path)


# param x_test: Data to predict on. Shape [n, nx, ny, channels]
test_path = "/con_data/heatmap_work/val_unet/"

test_images = glob.glob("test_path" + "*.jpg")
shuffle(test_images) 

test_images = test_images[0:5] # 

num_test_images = len(test_images) 
x_test = np.zeros((num_test_images,800,800,3))

for img_idx in range(num_test_images): 
  img_data = cv2.imread(test_images[img_idx]) 
  x_test[img_idx,:,:,:] = img_data


init = tf.global_variables_initializer()
with tf.Session() as sess:
  # Initialize variables
  sess.run(init)

  # Restore model weights from previously saved model
  restore(sess, model_path)

  y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
  prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

ipdb.set_trace()
print(prediction)


