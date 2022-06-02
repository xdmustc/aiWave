from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

clip_value = 1000000.


def quant(LLL, HLL_collection, LHL_collection, HHL_collection,
          LLH_collection, HLH_collection, LHH_collection, HHH_collection, s):
    with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):
        # HL_collection[0] = tf.round( s * tf.clip_by_value(HL_collection[0], -clip_value, clip_value) )  # 四舍五入取偶
        # LH_collection[0] = tf.round( s * tf.clip_by_value(LH_collection[0], -clip_value, clip_value) )
        # HH_collection[0] = tf.round( s * tf.clip_by_value(HH_collection[0], -clip_value, clip_value) )

        HLL_collection[0] = tf.round(s * tf.clip_by_value(HLL_collection[0], -clip_value, clip_value))
        LHL_collection[0] = tf.round(s * tf.clip_by_value(LHL_collection[0], -clip_value, clip_value))
        HHL_collection[0] = tf.round(s * tf.clip_by_value(HHL_collection[0], -clip_value, clip_value))
        LLH_collection[0] = tf.round(s * tf.clip_by_value(LLH_collection[0], -clip_value, clip_value))
        HLH_collection[0] = tf.round(s * tf.clip_by_value(HLH_collection[0], -clip_value, clip_value))
        LHH_collection[0] = tf.round(s * tf.clip_by_value(LHH_collection[0], -clip_value, clip_value))
        HHH_collection[0] = tf.round(s * tf.clip_by_value(HHH_collection[0], -clip_value, clip_value))

        # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。

        HLL_collection[1] = tf.round(s * tf.clip_by_value(HLL_collection[1], -clip_value, clip_value))
        LHL_collection[1] = tf.round(s * tf.clip_by_value(LHL_collection[1], -clip_value, clip_value))
        HHL_collection[1] = tf.round(s * tf.clip_by_value(HHL_collection[1], -clip_value, clip_value))
        LLH_collection[1] = tf.round(s * tf.clip_by_value(LLH_collection[1], -clip_value, clip_value))
        HLH_collection[1] = tf.round(s * tf.clip_by_value(HLH_collection[1], -clip_value, clip_value))
        LHH_collection[1] = tf.round(s * tf.clip_by_value(LHH_collection[1], -clip_value, clip_value))
        HHH_collection[1] = tf.round(s * tf.clip_by_value(HHH_collection[1], -clip_value, clip_value))

        # HLL_collection[2] = tf.round(s * tf.clip_by_value(HLL_collection[2], -clip_value, clip_value))
        # LHL_collection[2] = tf.round(s * tf.clip_by_value(LHL_collection[2], -clip_value, clip_value))
        # HHL_collection[2] = tf.round(s * tf.clip_by_value(HHL_collection[2], -clip_value, clip_value))
        # LLH_collection[2] = tf.round(s * tf.clip_by_value(LLH_collection[2], -clip_value, clip_value))
        # HLH_collection[2] = tf.round(s * tf.clip_by_value(HLH_collection[2], -clip_value, clip_value))
        # LHH_collection[2] = tf.round(s * tf.clip_by_value(LHH_collection[2], -clip_value, clip_value))
        # HHH_collection[2] = tf.round(s * tf.clip_by_value(HHH_collection[2], -clip_value, clip_value))
        #
        # HLL_collection[3] = tf.round(s * tf.clip_by_value(HLL_collection[3], -clip_value, clip_value))
        # LHL_collection[3] = tf.round(s * tf.clip_by_value(LHL_collection[3], -clip_value, clip_value))
        # HHL_collection[3] = tf.round(s * tf.clip_by_value(HHL_collection[3], -clip_value, clip_value))
        # LLH_collection[3] = tf.round(s * tf.clip_by_value(LLH_collection[3], -clip_value, clip_value))
        # HLH_collection[3] = tf.round(s * tf.clip_by_value(HLH_collection[3], -clip_value, clip_value))
        # LHH_collection[3] = tf.round(s * tf.clip_by_value(LHH_collection[3], -clip_value, clip_value))
        # HHH_collection[3] = tf.round(s * tf.clip_by_value(HHH_collection[3], -clip_value, clip_value))

        LLL = tf.round(s * tf.clip_by_value(LLL, -clip_value, clip_value))

    return LLL, HLL_collection, LHL_collection, HHL_collection, LLH_collection, HLH_collection, LHH_collection, HHH_collection


def de_quant(LLL, HLL_collection, LHL_collection, HHL_collection, LLH_collection, HLH_collection, LHH_collection,
             HHH_collection, s):
    HLL_collection[0] = HLL_collection[0] / s
    LHL_collection[0] = LHL_collection[0] / s
    HHL_collection[0] = HHL_collection[0] / s
    LLH_collection[0] = LLH_collection[0] / s
    HLH_collection[0] = HLH_collection[0] / s
    LHH_collection[0] = LHH_collection[0] / s
    HHH_collection[0] = HHH_collection[0] / s

    HLL_collection[1] = HLL_collection[1] / s
    LHL_collection[1] = LHL_collection[1] / s
    HHL_collection[1] = HHL_collection[1] / s
    LLH_collection[1] = LLH_collection[1] / s
    HLH_collection[1] = HLH_collection[1] / s
    LHH_collection[1] = LHH_collection[1] / s
    HHH_collection[1] = HHH_collection[1] / s
    #
    # HLL_collection[2] = HLL_collection[2] / s
    # LHL_collection[2] = LHL_collection[2] / s
    # HHL_collection[2] = HHL_collection[2] / s
    # LLH_collection[2] = LLH_collection[2] / s
    # HLH_collection[2] = HLH_collection[2] / s
    # LHH_collection[2] = LHH_collection[2] / s
    # HHH_collection[2] = HHH_collection[2] / s
    #
    # HLL_collection[3] = HLL_collection[3] / s
    # LHL_collection[3] = LHL_collection[3] / s
    # HHL_collection[3] = HHL_collection[3] / s
    # LLH_collection[3] = LLH_collection[3] / s
    # HLH_collection[3] = HLH_collection[3] / s
    # LHH_collection[3] = LHH_collection[3] / s
    # HHH_collection[3] = HHH_collection[3] / s

    LLL = LLL / s

    return LLL, HLL_collection, LHL_collection, HHL_collection, LLH_collection, HLH_collection, LHH_collection, HHH_collection
