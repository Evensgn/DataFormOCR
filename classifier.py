import numpy as np
import tensorflow as tf
import pickle
import cv2

## disable tensorflow warning output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GRAY_SCALE_RANGE = 255
NUM_ROW = 28
NUM_COLUMN = 28

num_pixel = NUM_ROW * NUM_COLUMN

def weight_variable(shape, name_):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name_)

def bias_variable(shape, name_):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name_)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

CONV_L = [1, 32, 64]
CONV_LAYERS = len(CONV_L) - 1

DENSE_L = [7 * 7 * 64, 1024, 10]
DENSE_LAYERS = len(DENSE_L) - 1

learning_rate = 5e-4
iterations = 100 # 30000
batch_size = 50
regular_lambda = 1e-6
drop_keep_prob = 0.5

x = tf.placeholder("float", [None, num_pixel])

conv_W = list(range(CONV_LAYERS + 1))
conv_b = list(range(CONV_LAYERS + 1))
for i in range(CONV_LAYERS):
    conv_W[i + 1] = weight_variable([5, 5, CONV_L[i], CONV_L[i + 1]], 'conv_W' + str(i + 1))
    conv_b[i + 1] = bias_variable([CONV_L[i + 1]], 'conv_b' + str(i + 1))

conv_yt = list(range(CONV_LAYERS + 1))
conv_yt_pool = list(range(CONV_LAYERS + 1))
conv_yt_pool[0] = tf.reshape(x, [-1, 28, 28, CONV_L[0]])
for i in range(CONV_LAYERS):
    conv_yt[i + 1] = tf.nn.relu(conv2d(conv_yt_pool[i], conv_W[i + 1]) + conv_b[i + 1])
    conv_yt_pool[i + 1] = max_pool_2x2(conv_yt[i + 1])

dense_W = list(range(DENSE_LAYERS + 1))
dense_b = list(range(DENSE_LAYERS + 1))
for i in range(DENSE_LAYERS):
    dense_W[i + 1] = weight_variable([DENSE_L[i], DENSE_L[i + 1]], 'dense_W' + str(i + 1))
    dense_b[i + 1] = bias_variable([DENSE_L[i + 1]], 'dense_b' + str(i + 1))

dense_yt = list(range(DENSE_LAYERS + 1))
dense_yt[0] = tf.reshape(conv_yt_pool[CONV_LAYERS], [-1, DENSE_L[0]])
for i in range(DENSE_LAYERS):
    if i == DENSE_LAYERS - 1:
        dense_yt[i + 1] = tf.matmul(dense_yt[i], dense_W[i + 1]) + dense_b[i + 1]
    else:
        dense_yt[i + 1] = tf.nn.relu(tf.matmul(dense_yt[i], dense_W[i + 1]) + dense_b[i + 1])

y = tf.nn.softmax(dense_yt[DENSE_LAYERS])

prediction = tf.argmax(y, 1)

saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, './clf.ckpt')
print('Classifier initialization.')

def show_large_img(img):
    f = lambda i, j: img[i // 10, j // 10]
    imglarge = np.fromfunction(f, (img.shape[0] * 10, img.shape[1] * 10), dtype = np.uint8)        
    cv2.imshow('Image', imglarge)
    cv2.waitKey(0)

def classify(images):
	images = np.array(images)
	images.resize(images.size // num_pixel, num_pixel)
	images = images / GRAY_SCALE_RANGE

	ret = sess.run(prediction, feed_dict = {x: images})
	return ret