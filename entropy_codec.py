from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pixelcnn_3D
import pixelcnn_3D_context


def codec(LLL1, HLL_collection, LHL_collection, HHL_collection, LLH_collection, HLH_collection, LHH_collection,
          HHH_collection, c_HLL, c_LHL, c_HHL, c_LLH, c_HLH, c_LHH, c_HHH, static_QP):
    with tf.variable_scope('LLL1'):
        ce_loss = pixelcnn_3D.mask_3D_layer(LLL1, static_QP)

    for j in range(2):
        i = 2 - 1 - j

        with tf.variable_scope('HLL' + str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(HLL_collection[i], static_QP, c_HLL[j]) + ce_loss
        with tf.variable_scope('LHL' + str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(LHL_collection[i], static_QP, c_LHL[j]) + ce_loss
        with tf.variable_scope('HHL' + str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(HHL_collection[i], static_QP, c_HHL[j]) + ce_loss
        with tf.variable_scope('LLH' + str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(LLH_collection[i], static_QP, c_LLH[j]) + ce_loss
        with tf.variable_scope('HLH' + str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(HLH_collection[i], static_QP, c_HLH[j]) + ce_loss
        with tf.variable_scope('LHH' + str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(LHH_collection[i], static_QP, c_LHH[j]) + ce_loss
        with tf.variable_scope('HHH' + str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(HHH_collection[i], static_QP, c_HHH[j]) + ce_loss

    return ce_loss / (64. * 64. * 64.)
