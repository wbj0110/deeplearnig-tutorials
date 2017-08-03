import tensorflow as tf
import numpy as np
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import load_image

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
num = 20
imgs = []
for i in range(1, num + 1):
    imgs.append(np.asarray(load_image("%s/cnn_dataset_mini/miku/%s.jpg" % (SCRIPT_PATH, i))))
for i in range(1, num + 1):
    imgs.append(np.asarray(load_image("%s/cnn_dataset_mini/no-miku/%s.jpg" % (SCRIPT_PATH, i))))
imgs = np.array(imgs)
y_data = np.r_[np.c_[np.ones(num), np.zeros(num)],np.c_[np.zeros(num), np.ones(num)]]
print(imgs.shape)
print(y_data.shape)

x_test = []
for i in range(1, 11):
    x_test.append(np.asarray(load_image("%s/cnn_dataset_mini/test-set/%s.jpg" % (SCRIPT_PATH, i))))
x_test =  np.array(x_test)
y_test = np.r_[np.c_[np.ones(5), np.zeros(5)],np.c_[np.zeros(5), np.ones(5)]]
print(x_test.shape)
print(y_test.shape)
#output

#(10, 100, 100, 3)

#(10, 2)

# Building convolutional network

network = input_data(shape=[None, 100, 100, 3], name='input')
network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 128, 5, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 512, activation='relu')
#network = dropout(network, 0.8)

network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.00001,
                     loss='categorical_crossentropy', name='target')


# Training

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': imgs}, {'target': y_data}, n_epoch=500,
           validation_set=({'input': x_test}, {'target': y_test}),
           snapshot_step=100,show_metric=True, run_id='convnet_miku')
model.save('miku_model.tflearn')





