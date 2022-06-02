from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import basic_DL_op


def lstm_logic(x, c):
    i = tf.sigmoid(x)
    f = tf.sigmoid(x)
    o = tf.sigmoid(x)
    g = tf.tanh(x)

    c = f * c + i * g

    h = o * tf.tanh(c)

    return c, h


def lstm_layer(x, h, c, in_num, out_num):
    # the first layer: input

    w = basic_DL_op.weight_variable('conv1', [3, 3, 3, in_num, out_num], 0.01)

    x = basic_DL_op.conv3d(x, w)

    # the first layer: state

    w = basic_DL_op.weight_variable('conv2', [3, 3, 3, out_num, out_num], 0.01)

    h = basic_DL_op.conv3d(h, w)

    b = basic_DL_op.bias_variable('bias', [out_num])

    c, h = lstm_logic(x + h + b, c)

    return c, h


def context_single_band(x1, h1, c1, h2, c2, h3, c3, bit_map):
    with tf.variable_scope('LSTM_' + str(1)):
        c1, h1 = lstm_layer(x1, h1, c1, 1, int(32 * bit_map))

    with tf.variable_scope('LSTM_' + str(2)):
        c2, h2 = lstm_layer(h1, h2, c2, int(32 * bit_map), int(32 * bit_map))

    with tf.variable_scope('LSTM_' + str(3)):
        c3, h3 = lstm_layer(h2, h3, c3, int(32 * bit_map), 1)

    return h1, c1, h2, c2, h3, c3


def context_single_band_reuse(x1, h1, c1, h2, c2, h3, c3, bit_map):
    with tf.variable_scope('LSTM_' + str(1), reuse=True):
        c1, h1 = lstm_layer(x1, h1, c1, 1, int(32 * bit_map))

    with tf.variable_scope('LSTM_' + str(2), reuse=True):
        c2, h2 = lstm_layer(h1, h2, c2, int(32 * bit_map), int(32 * bit_map))

    with tf.variable_scope('LSTM_' + str(3), reuse=True):
        c3, h3 = lstm_layer(h2, h3, c3, int(32 * bit_map), 1)

    return h1, c1, h2, c2, h3, c3


def deconv_layer(x):
    x_shape = x.get_shape().as_list()

    kernel = basic_DL_op.weight_variable('deconv', [3, 3, 3, x_shape[4], x_shape[4]], 0.01)

    x = tf.nn.conv3d_transpose(x, kernel,
                               output_shape=[x_shape[0], int(x_shape[1] * 2), int(x_shape[2] * 2), int(x_shape[3] * 2),
                                             x_shape[4]], strides=[1, 2, 2, 2, 1], padding="SAME")

    return x


def context_all(LLL3, HLL_collection, LHL_collection, HHL_collection, LLH_collection, HLH_collection, LHH_collection,
                HHH_collection, static_QP, bit_map):
    c_LLH = []
    c_HLL = []
    c_LHL = []
    c_HLH = []
    c_HHL = []
    c_LHH = []
    c_HHH = []
    x_shape = LLL3.get_shape().as_list()

    h1 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[3], int(32 * bit_map)], dtype=tf.float32)
    c1 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[3], int(32 * bit_map)], dtype=tf.float32)

    h2 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[3], int(32 * bit_map)], dtype=tf.float32)
    c2 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[3], int(32 * bit_map)], dtype=tf.float32)

    h3 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[3], 1], dtype=tf.float32)
    c3 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[3], 1], dtype=tf.float32)

    h1, c1, h2, c2, h3, c3 = context_single_band(LLL3 / static_QP, h1, c1, h2, c2, h3, c3, bit_map)

    c_HLL.append(h3)

    for j in range(2):
        i = 2 - 1 - j

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(HLL_collection[i] / static_QP, h1, c1, h2, c2, h3, c3,
                                                           bit_map)

        c_LHL.append(h3)

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(LHL_collection[i] / static_QP, h1, c1, h2, c2, h3, c3,
                                                           bit_map)

        c_HHL.append(h3)

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(HHL_collection[i] / static_QP, h1, c1, h2, c2, h3, c3,
                                                           bit_map)

        c_LLH.append(h3)

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(LLH_collection[i] / static_QP, h1, c1, h2, c2, h3, c3,
                                                           bit_map)

        c_HLH.append(h3)

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(HLH_collection[i] / static_QP, h1, c1, h2, c2, h3, c3,
                                                           bit_map)

        c_LHH.append(h3)

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(LHH_collection[i] / static_QP, h1, c1, h2, c2, h3, c3,
                                                           bit_map)

        c_HHH.append(h3)

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(HHH_collection[i] / static_QP, h1, c1, h2, c2, h3, c3,
                                                           bit_map)

        with tf.variable_scope('Deconv_h1' + str(j)):
            h1 = deconv_layer(h1)
        with tf.variable_scope('Deconv_c1' + str(j)):
            c1 = deconv_layer(c1)
        with tf.variable_scope('Deconv_h2' + str(j)):
            h2 = deconv_layer(h2)
        with tf.variable_scope('Deconv_c2' + str(j)):
            c2 = deconv_layer(c2)
        with tf.variable_scope('Deconv_h3' + str(j)):
            h3 = deconv_layer(h3)
        with tf.variable_scope('Deconv_c3' + str(j)):
            c3 = deconv_layer(c3)

        c_HLL.append(h3)

    return c_HLL, c_LHL, c_HHL, c_LLH, c_HLH, c_LHH, c_HHH
