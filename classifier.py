import numpy as np
import tensorflow as tf
import pickle

## disable tensorflow warning output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GRAY_SCALE_RANGE = 255
NUM_ROW = 28
NUM_COLUMN = 28

num_pixel = NUM_ROW * NUM_COLUMN

CONV_L = [1, 32, 64]
CONV_LAYERS = len(CONV_L) - 1

DENSE_L = [7 * 7 * 64, 1024, 10]
DENSE_LAYERS = len(DENSE_L) - 1

learning_rate = 5e-4
iterations = 30000
batch_size = 50
regular_lambda = 1e-6

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

x = tf.placeholder("float", [None, num_pixel])

conv_W = list(range(CONV_LAYERS + 1))
conv_b = list(range(CONV_LAYERS + 1))
for i in range(CONV_LAYERS):
    conv_W[i + 1] = weight_variable([5, 5, CONV_L[i], CONV_L[i + 1]])
    conv_b[i + 1] = bias_variable([CONV_L[i + 1]])

conv_yt = list(range(CONV_LAYERS + 1))
conv_yt_pool = list(range(CONV_LAYERS + 1))
conv_yt_pool[0] = tf.reshape(x, [-1, 28, 28, CONV_L[0]])
for i in range(CONV_LAYERS):
    conv_yt[i + 1] = tf.nn.relu(conv2d(conv_yt_pool[i], conv_W[i + 1]) + conv_b[i + 1])
    conv_yt_pool[i + 1] = max_pool_2x2(conv_yt[i + 1])

dense_W = list(range(DENSE_LAYERS + 1))
dense_b = list(range(DENSE_LAYERS + 1))
for i in range(DENSE_LAYERS):
    dense_W[i + 1] = weight_variable([DENSE_L[i], DENSE_L[i + 1]])
    dense_b[i + 1] = bias_variable([DENSE_L[i + 1]])

dense_yt = list(range(DENSE_LAYERS + 1))
dense_yt[0] = tf.reshape(conv_yt_pool[CONV_LAYERS], [-1, DENSE_L[0]])
for i in range(DENSE_LAYERS):
    if i == DENSE_LAYERS - 1:
        dense_yt[i + 1] = tf.matmul(dense_yt[i], dense_W[i + 1]) + dense_b[i + 1]
    else:
        dense_yt[i + 1] = tf.nn.relu(tf.matmul(dense_yt[i], dense_W[i + 1]) + dense_b[i + 1])

y = tf.nn.softmax(dense_yt[DENSE_LAYERS])
init = tf.global_variables_initializer()

prediction = tf.argmax(y, 1)

sess = tf.Session()
sess.run(init)
print('Classifier initialization.')

def load_clf(clf_filename):
	print('Loading classifier parameters from file \'' + clf_filename + '\' ...')
	with open(clf_filename, 'rb') as f:
		for i in range(CONV_LAYERS):
			conv_W[i + 1] = pickle.load(f)
			conv_b[i + 1] = pickle.load(f)
		for i in range(DENSE_LAYERS):
			dense_W[i + 1] = pickle.load(f)
			dense_b[i + 1] = pickle.load(f)
	print('Loading complete.')

def classify(img_digits):
	images = []
	for img_digit in img_digits:
		
	images = np.array(images)
	images.resize(images.size // num_pixel, num_pixel)
	images = images / GRAY_SCALE_RANGE
	ret = sess.run(prediction, feed_dict = {x: images})
	return ret