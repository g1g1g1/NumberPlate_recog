'''
Code source: https://github.com/matthewearl/deep-anpr
'''

__all__ = (
    'get_training_model',
    'get_detect_model',
    'WINDOW_SHAPE',
)

import tensorflow as tf
import common

WINDOW_SHAPE = (64, 128)

# Utility functions

# weight variable 초기화 방식을 trauncated_normal -> xavier init
def weight_variable(shape):
  #initial = tf.truncated_normal(shape, stddev=0.1)
  #return tf.Variable(initial)
  return tf.get_variable(shape=shape, initializer=tf.contrib.layers.xavier_initializer())

# bias variable 초기화 방식을 0.1 -> 0으로
def bias_variable(shape):
  #initial = tf.constant(0.1, shape=shape)
  #return tf.Variable(initial)
  return tf.Variable(tf.zeros(shape))

def conv2d(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


# 각 레이어 사이에 dropout 추가
def convolutional_layers():
    """
    Get the convolutional layers of the model.
    """
    x = tf.placeholder(tf.float32, [None, None, None])
    kp = tf.placeholder(tf.float32)  # dropout위한 placeholder

    # First layer
    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])
    x_expanded = tf.expand_dims(x, 3)
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # dropout 추가
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob=kp)

    # Second layer
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    # dropout 추가
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob=kp)

    # Third layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    # dropout 추가
    h_pool3 = tf.nn.dropout(h_pool3, keep_prob=kp)

    return x, h_pool3, [W_conv1, b_conv1,
                        W_conv2, b_conv2,
                        W_conv3, b_conv3] , kp

# 각 레이어 사이에 dropout 추가
def get_training_model():
    """
    The training model acts on a batch of 128x64 windows, and outputs a (1 +
    7 * len(common.CHARS) vector, `v`. `v[0]` is the probability that a plate is
    fully within the image and is at the correct scale.
    
    `v[1 + i * len(common.CHARS) + c]` is the probability that the `i`'th
    character is `c`.
    """

    # training 시에는 dropout 확률을 0.7로 세팅
    x, conv_layer, conv_vars, kp = convolutional_layers()
    
    # Densely connected layer
    W_fc1 = weight_variable([32 * 8 * 128, 2048])
    b_fc1 = bias_variable([2048])

    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # dropout 추가
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob=kp)

    # Output layer
    W_fc2 = weight_variable([2048, 1 + 7 * len(common.CHARS)])
    b_fc2 = bias_variable([1 + 7 * len(common.CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return (x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2], kp)


# 각 레이어 사이에 dropout 추가
def get_detect_model():
    """
    The same as the training model, except it acts on an arbitrarily sized
    input, and slides the 128x64 window across the image in 8x8 strides.
    The output is of the form `v`, where `v[i, j]` is equivalent to the output
    of the training model, for the window at coordinates `(8 * i, 4 * j)`.
    """

    x, conv_layer, conv_vars, kp = convolutional_layers()
    
    # Fourth layer
    W_fc1 = weight_variable([8 * 32 * 128, 2048])
    W_conv1 = tf.reshape(W_fc1, [8,  32, 128, 2048])
    b_fc1 = bias_variable([2048])
    h_conv1 = tf.nn.relu(conv2d(conv_layer, W_conv1,
                                stride=(1, 1), padding="VALID") + b_fc1)

    #dropout 추가
    h_conv1 = tf.nn.dropout(h_conv1, keep_prob=kp)

    # Fifth layer
    W_fc2 = weight_variable([2048, 1 + 7 * len(common.CHARS)])
    W_conv2 = tf.reshape(W_fc2, [1, 1, 2048, 1 + 7 * len(common.CHARS)])
    b_fc2 = bias_variable([1 + 7 * len(common.CHARS)])
    h_conv2 = conv2d(h_conv1, W_conv2) + b_fc2

    return (x, h_conv2, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2], kp)
