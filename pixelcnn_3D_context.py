from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import basic_DL_op
import numpy as np


def mask_3D_resiBlock(x, filter_nums):
    w = basic_DL_op.weight_variable('conv1', [3, 3, 3, filter_nums, filter_nums], 0.01)

    mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]
             ],
            [[1, 1, 1], [1, 1, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]
             ]
            ]

    mask = tf.reshape(mask, shape=[3, 3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, 1, filter_nums, filter_nums])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias1', [filter_nums])

    c = basic_DL_op.conv3d(x, w) + b

    c = tf.nn.relu(c)

    w = basic_DL_op.weight_variable('conv2', [3, 3, 3, filter_nums, filter_nums], 0.01)

    w = w * mask

    b = basic_DL_op.bias_variable('bias2', [filter_nums])

    c = basic_DL_op.conv3d(c, w) + b

    return x + c


def resiBlock_3D(x, filter_nums):
    w = basic_DL_op.weight_variable('conv1_c', [3, 3, 3, filter_nums, filter_nums], 0.01)

    b = basic_DL_op.bias_variable('bias1_c', [filter_nums])

    c = basic_DL_op.conv3d(x, w) + b

    c = tf.nn.relu(c)

    w = basic_DL_op.weight_variable('conv2_c', [3, 3, 3, filter_nums, filter_nums], 0.01)

    b = basic_DL_op.bias_variable('bias2_c', [filter_nums])

    c = basic_DL_op.conv3d(c, w) + b

    return x + c


def mask_3D_layer(x, static_QP, context, out_dim=128, resi_num=2, para_num=58):
    x = x / static_QP

    label = x

    # creat mask convolution kernels

    w = basic_DL_op.weight_variable('conv1', [3, 3, 3, 1, out_dim], 0.01)
    w_c = basic_DL_op.weight_variable('conv1_c', [3, 3, 3, 1, out_dim], 0.01)

    mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]
             ],
            [[1, 1, 1], [1, 0, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]
             ]
            ]


    mask = tf.reshape(mask, shape=[3, 3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, 1, 1, out_dim])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias1', [out_dim])
    b_c = basic_DL_op.bias_variable('bias1_c', [out_dim])

    x = basic_DL_op.conv3d(x, w) + b
    long_context = basic_DL_op.conv3d(context, w_c) + b_c

    conv1 = x
    x = x + long_context

    for i in range(resi_num):
        with tf.variable_scope('resi_block' + str(i)):
            x = mask_3D_resiBlock(x, out_dim)
            long_context = resiBlock_3D(long_context, out_dim)
            x = x + long_context

    x = conv1 + x

    w = basic_DL_op.weight_variable('conv2', [3, 3, 3, out_dim, out_dim], 0.01)

    mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]
             ],
            [[1, 1, 1], [1, 1, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]
             ]
            ]

    mask = tf.reshape(mask, shape=[3, 3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, 1, out_dim, out_dim])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias2', [out_dim])

    x = basic_DL_op.conv3d(x, w) + b

    x = tf.nn.relu(x)

    ################# convs: 1x1, relu/linear

    w = basic_DL_op.weight_variable('conv3', [1, 1, 1, out_dim, out_dim], 0.01)

    b = basic_DL_op.bias_variable('bias3', [out_dim])

    x = basic_DL_op.conv3d(x, w) + b

    x = tf.nn.relu(x)

    w = basic_DL_op.weight_variable('conv4', [1, 1, 1, out_dim, out_dim], 0.01)

    b = basic_DL_op.bias_variable('bias4', [out_dim])

    x = basic_DL_op.conv3d(x, w) + b

    x = tf.nn.relu(x)

    w = basic_DL_op.weight_variable('conv5', [1, 1, 1, out_dim, para_num], 0.01)

    b = basic_DL_op.bias_variable('bias5', [para_num])

    x = basic_DL_op.conv3d(x, w) + b

    ################# cal the cdf with the output params

    h = tf.nn.softplus(x[:, :, :, :, 0:33])
    b = x[:, :, :, :, 33:46]
    a = tf.tanh(x[:, :, :, :, 46:58])

    lower = label - 0.5 / static_QP
    high = label + 0.5 / static_QP

    lower = cal_cdf(lower, h, b, a)
    high = cal_cdf(high, h, b, a)

    prob = tf.maximum((high - lower), 1e-9)

    cross_entropy = -tf.reduce_sum(tf.log(prob)) / np.log(2)

    return cross_entropy


def cal_cdf(logits, h, b, a):
    shape = logits.get_shape().as_list()
    logits = tf.reshape(logits, [shape[0], shape[1], shape[2], shape[3], shape[4], 1])

    logits = tf.matmul(tf.reshape(h[:, :, :, :, 0:3], [shape[0], shape[1], shape[2], shape[3], 3, 1]), logits)
    logits = logits + tf.reshape(b[:, :, :, :, 0:3], [shape[0], shape[1], shape[2], shape[3], 3, 1])
    logits = logits + tf.reshape(a[:, :, :, :, 0:3], [shape[0], shape[1], shape[2], shape[3], 3, 1]) * tf.tanh(logits)

    logits = tf.matmul(tf.reshape(h[:, :, :, :, 3:12], [shape[0], shape[1], shape[2], shape[3], 3, 3]), logits)
    logits = logits + tf.reshape(b[:, :, :, :, 3:6], [shape[0], shape[1], shape[2], shape[3], 3, 1])
    logits = logits + tf.reshape(a[:, :, :, :, 3:6], [shape[0], shape[1], shape[2], shape[3], 3, 1]) * tf.tanh(logits)

    logits = tf.matmul(tf.reshape(h[:, :, :, :, 12:21], [shape[0], shape[1], shape[2], shape[3], 3, 3]), logits)
    logits = logits + tf.reshape(b[:, :, :, :, 6:9], [shape[0], shape[1], shape[2], shape[3], 3, 1])
    logits = logits + tf.reshape(a[:, :, :, :, 6:9], [shape[0], shape[1], shape[2], shape[3], 3, 1]) * tf.tanh(logits)

    logits = tf.matmul(tf.reshape(h[:, :, :, :, 21:30], [shape[0], shape[1], shape[2], shape[3], 3, 3]), logits)
    logits = logits + tf.reshape(b[:, :, :, :, 9:12], [shape[0], shape[1], shape[2], shape[3], 3, 1])
    logits = logits + tf.reshape(a[:, :, :, :, 9:12], [shape[0], shape[1], shape[2], shape[3], 3, 1]) * tf.tanh(logits)

    logits = tf.matmul(tf.reshape(h[:, :, :, :, 30:33], [shape[0], shape[1], shape[2], shape[3], 1, 3]), logits)
    logits = logits + tf.reshape(b[:, :, :, :, 12:13], [shape[0], shape[1], shape[2], shape[3], 1, 1])

    logits = tf.sigmoid(logits)
    logits = tf.reshape(logits, [shape[0], shape[1], shape[2], shape[3], shape[4]])

    return logits
