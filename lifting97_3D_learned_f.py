from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import basic_DL_op

######################## lifting 97 forward and inverse transform

lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781,
                 1.149604398860241]  # bior4.4
affine_eps = 0.
trainable_set = True


def p_block1(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv1', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias1', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv2', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias2', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv3', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias3', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv4', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias4', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv5', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias5', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block2(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv21', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias21', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv22', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias22', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv23', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias23', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv24', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias24', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv25', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias25', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block3(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv31', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias31', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv32', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias32', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv33', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias33', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv34', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias34', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv35', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias35', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block4(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv41', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias41', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv42', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias42', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv43', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias43', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv44', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias44', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv45', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias45', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block5(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv51', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias51', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv52', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias52', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv53', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias53', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv54', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias54', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv55', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias55', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block6(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv61', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias61', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv62', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias62', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv63', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias63', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv64', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias64', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv65', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias65', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block7(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv71', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias71', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv72', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias72', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv73', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias73', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv74', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias74', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv75', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias75', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block8(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv81', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias81', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv82', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias82', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv83', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias83', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv84', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias84', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv85', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias85', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block9(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv91', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias91', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv92', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias92', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv93', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias93', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv94', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias94', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv95', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias95', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block10(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv101', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias101', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv102', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias102', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv103', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias103', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv104', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias104', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv105', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias105', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block11(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv111', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias111', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv112', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias112', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv113', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias113', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv114', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias114', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv115', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias115', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block12(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv121', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias121', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv122', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias122', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv123', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias123', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv124', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias124', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv125', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias125', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block13(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv131', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias131', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv132', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias132', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv133', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias133', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv134', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias134', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv135', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias135', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out

def p_block14(x):
    #  x =1,42,64,64,1
    w = basic_DL_op.weight_variable('conv141', [3, 3, 3, 1, 16], 0.01)
    b = basic_DL_op.bias_variable('bias141', [16])
    tmp = basic_DL_op.conv3d_pad(x, w) + b

    conv1 = tmp

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv142', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias142', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = tf.nn.relu(tmp)
    w = basic_DL_op.weight_variable('conv143', [3, 3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias143', [16])
    tmp = basic_DL_op.conv3d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv144', [3, 3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias144', [1])
    out1 = basic_DL_op.conv3d_pad(tmp, w) + b

    w = basic_DL_op.weight_variable1('conv145', [3, 3, 3, 16, 1])
    b = basic_DL_op.bias_variable1('bias145', [1])
    out2 = basic_DL_op.conv3d_pad1(tmp, w) + b

    out = tf.concat([out2, out1], 4)

    return out


def lifting97_forward1(L, H):

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")
    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_10', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block1(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_11', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block1(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift+ skip
    L = L * scale

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_12', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block2(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_13', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block2(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift + skip
    L = L * scale

    # scaling step

    n_h = tf.get_variable('n_h1', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H * n_h

    n_l = tf.get_variable('n_l1', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L * n_l

    return L, H

def lifting97_forward2(L, H):

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")  # tf.pad()函数主要是对张量在各个维度上进行填充
    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])
# 3,1,1,1

    w = tf.get_variable('cdf97_20', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block3(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_21', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block3(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift+ skip
    L = L * scale

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_22', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block4(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_23', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block4(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift + skip
    L = L * scale

    # scaling step

    n_h = tf.get_variable('n_h2', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H * n_h

    n_l = tf.get_variable('n_l2', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L * n_l

    return L, H

def lifting97_forward3(L, H):

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")
    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_30', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block5(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_31', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block5(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift+ skip
    L = L * scale

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_32', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block6(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_33', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block6(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift + skip
    L = L * scale

    # scaling step

    n_h = tf.get_variable('n_h3', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H * n_h

    n_l = tf.get_variable('n_l3', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L * n_l

    return L, H

def lifting97_forward4(L, H):

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")
    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_40', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block7(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_41', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block7(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift+ skip
    L = L * scale

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_42', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block8(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_43', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block8(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift + skip
    L = L * scale

    # scaling step

    n_h = tf.get_variable('n_h4', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H * n_h

    n_l = tf.get_variable('n_l4', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L * n_l

    return L, H

def lifting97_forward5(L, H):

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")
    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_50', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block9(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_51', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block9(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift+ skip
    L = L * scale

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_52', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block10(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_53', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block10(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift + skip
    L = L * scale

    # scaling step

    n_h = tf.get_variable('n_h5', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H * n_h

    n_l = tf.get_variable('n_l5', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L * n_l

    return L, H

def lifting97_forward6(L, H):

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")
    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_60', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block11(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_61', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block11(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift+ skip
    L = L * scale

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_62', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block12(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_63', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block12(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift + skip
    L = L * scale

    # scaling step

    n_h = tf.get_variable('n_h6', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H * n_h

    n_l = tf.get_variable('n_l6', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L * n_l

    return L, H

def lifting97_forward7(L, H):

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")
    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_70', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block13(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale


    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_71', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block13(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift+ skip
    L = L * scale

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_72', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block14(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H + shift + skip
    H = H * scale

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_73', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block14(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L + shift + skip
    L = L * scale

    # scaling step

    n_h = tf.get_variable('n_h7', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H * n_h

    n_l = tf.get_variable('n_l7', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L * n_l

    return L, H


def lifting97_inverse7(L, H):

    # scaling step

    n_h = tf.get_variable('n_h7', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H / n_h

    n_l = tf.get_variable('n_l7', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L / n_l

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_73', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block14(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip


    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_72', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block14(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    # first lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_71', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block13(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_70', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block13(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)
    H = H / scale
    H = H - shift - skip

    return L, H

def lifting97_inverse6(L, H):
    # scaling step

    n_h = tf.get_variable('n_h6', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H / n_h

    n_l = tf.get_variable('n_l6', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L / n_l

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_63', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block12(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_62', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block12(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    # first lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_61', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block11(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_60', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block11(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    return L, H

def lifting97_inverse5(L, H):
    # scaling step

    n_h = tf.get_variable('n_h5', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H / n_h

    n_l = tf.get_variable('n_l5', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L / n_l

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_53', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block10(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_52', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block10(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    # first lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_51', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block9(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_50', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block9(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    return L, H

def lifting97_inverse4(L, H):
    # scaling step

    n_h = tf.get_variable('n_h4', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H / n_h

    n_l = tf.get_variable('n_l4', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L / n_l

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_43', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block8(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_42', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block8(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    # first lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_41', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block7(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_40', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block7(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    return L, H

def lifting97_inverse3(L, H):
    # scaling step

    n_h = tf.get_variable('n_h3', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H / n_h

    n_l = tf.get_variable('n_l3', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L / n_l

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_33', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block6(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_32', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block6(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    # first lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_31', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block5(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_30', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block5(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    return L, H

def lifting97_inverse2(L, H):
    # scaling step

    n_h = tf.get_variable('n_h2', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H / n_h

    n_l = tf.get_variable('n_l2', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L / n_l

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_23', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block4(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_22', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block4(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    # first lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_21', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block3(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip


    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_20', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block3(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    return L, H

def lifting97_inverse1(L, H):
    # scaling step

    n_h = tf.get_variable('n_h1', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h

    H = H / n_h

    n_l = tf.get_variable('n_l1', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l

    L = L / n_l

    # second lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_13', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block2(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_12', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block2(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    # first lifting step

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_11', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block1(skip / 256.) * 256.

    scale, shift = tf.split(H_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    L = L / scale
    L = L - shift - skip

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1, 1])

    w = tf.get_variable('cdf97_10', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv3d(tmp, w, strides=[1, 1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block1(skip / 256.) * 256.

    scale, shift = tf.split(L_net, 2, 4)
    scale = (tf.sigmoid(scale + 2.) + affine_eps)

    H = H / scale
    H = H - shift - skip

    return L, H
