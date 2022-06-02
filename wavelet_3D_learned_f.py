from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import lifting97_3D_learned_f


############### one-level decomposition of lifting 97

def decomposition_depth(subband):
    subband = tf.transpose(subband, [0, 3, 2, 1, 4])
    subband_L = subband[:, 0::2, :, :, :]
    subband_H = subband[:, 1::2, :, :, :]
    subband_L, subband_H = lifting97_3D_learned_f.lifting97_forward4(subband_L, subband_H)
    subband_L = tf.transpose(subband_L, [0, 3, 2, 1, 4])
    subband_H = tf.transpose(subband_H, [0, 3, 2, 1, 4])
    return subband_L, subband_H


def reconstruct_depth(LLL, LLH):
    LLL = tf.transpose(LLL, [0, 3, 2, 1, 4])
    LLH = tf.transpose(LLH, [0, 3, 2, 1, 4])
    LLL, LLH = lifting97_3D_learned_f.lifting97_inverse4(LLL, LLH)
    LL = reconstruct_fun_depth(LLL, LLH)  # 在第二维concate就好啦
    LL = tf.transpose(LL, [0, 3, 2, 1, 4])
    return LL


def reconstruct_fun_depth(temp_L, temp_H):
    temp_L = tf.transpose(temp_L, [0, 3, 2, 1, 4])

    temp_H = tf.transpose(temp_H, [0, 3, 2, 1, 4])

    x_shape = temp_L.get_shape().as_list()

    x_n = x_shape[0]

    x_h = x_shape[1]

    x_w = x_shape[2]

    x_d = x_shape[3]

    x_c = x_shape[4]

    temp_L = tf.reshape(temp_L, [x_n, x_h * x_w * x_d, 1, x_c])

    temp_H = tf.reshape(temp_H, [x_n, x_h * x_w * x_d, 1, x_c])

    temp = tf.concat([temp_L, temp_H], 3)

    temp = tf.reshape(temp, [x_n, x_h, x_w, 2 * x_d, x_c])

    recon = tf.transpose(temp, [0, 3, 2, 1, 4])

    return recon


def decomposition(x):
    # img_3D[1, 128, 128, 84, 1]

    # step 1: for h
    L = x[:, 0::2, :, :, :]
    H = x[:, 1::2, :, :, :]

    L, H = lifting97_3D_learned_f.lifting97_forward1(L, H)

    # step 2: for w, L

    L = tf.transpose(L, [0, 2, 1, 3, 4])

    LL = L[:, 0::2, :, :, :]
    HL = L[:, 1::2, :, :, :]

    LL, HL = lifting97_3D_learned_f.lifting97_forward2(LL, HL)

    LL = tf.transpose(LL, [0, 2, 1, 3, 4])
    HL = tf.transpose(HL, [0, 2, 1, 3, 4])

    # step 2: for w, H

    H = tf.transpose(H, [0, 2, 1, 3, 4])

    LH = H[:, 0::2, :, :, :]
    HH = H[:, 1::2, :, :, :]

    LH, HH = lifting97_3D_learned_f.lifting97_forward3(LH, HH)

    LH = tf.transpose(LH, [0, 2, 1, 3, 4])
    HH = tf.transpose(HH, [0, 2, 1, 3, 4])

    subband = tf.transpose(LL, [0, 3, 2, 1, 4])
    subband_L = subband[:, 0::2, :, :, :]
    subband_H = subband[:, 1::2, :, :, :]
    subband_L, subband_H = lifting97_3D_learned_f.lifting97_forward4(subband_L, subband_H)
    LLL = tf.transpose(subband_L, [0, 3, 2, 1, 4])
    HLL = tf.transpose(subband_H, [0, 3, 2, 1, 4])

    subband = tf.transpose(HL, [0, 3, 2, 1, 4])
    subband_L = subband[:, 0::2, :, :, :]
    subband_H = subband[:, 1::2, :, :, :]
    subband_L, subband_H= lifting97_3D_learned_f.lifting97_forward5(subband_L, subband_H)
    LHL = tf.transpose(subband_L, [0, 3, 2, 1, 4])
    HHL = tf.transpose(subband_H, [0, 3, 2, 1, 4])

    subband = tf.transpose(LH, [0, 3, 2, 1, 4])
    subband_L = subband[:, 0::2, :, :, :]
    subband_H = subband[:, 1::2, :, :, :]
    subband_L, subband_H= lifting97_3D_learned_f.lifting97_forward6(subband_L, subband_H)
    LLH = tf.transpose(subband_L, [0, 3, 2, 1, 4])
    HLH = tf.transpose(subband_H, [0, 3, 2, 1, 4])

    subband = tf.transpose(HH, [0, 3, 2, 1, 4])
    subband_L = subband[:, 0::2, :, :, :]
    subband_H = subband[:, 1::2, :, :, :]
    subband_L, subband_H  = lifting97_3D_learned_f.lifting97_forward7(subband_L, subband_H)
    LHH = tf.transpose(subband_L, [0, 3, 2, 1, 4])
    HHH = tf.transpose(subband_H, [0, 3, 2, 1, 4])
    # scale = tf.transpose(scale, [0, 3, 2, 1, 4])

    return LLL, HLL, LHL, HHL, LLH, HLH, LHH, HHH


############### one-level reconstruction of lifting 97


def reconstruct_fun(up, bot):
    temp_L = tf.transpose(up, [0, 2, 1, 3, 4])
    temp_H = tf.transpose(bot, [0, 2, 1, 3, 4])

    x_shape = temp_L.get_shape().as_list()

    x_n = x_shape[0]
    x_h = x_shape[1]
    x_w = x_shape[2]
    x_d = x_shape[3]
    x_c = x_shape[4]
    temp_L = tf.reshape(temp_L, [x_n, x_h * x_w, 1, x_d, x_c])

    temp_H = tf.reshape(temp_H, [x_n, x_h * x_w, 1, x_d, x_c])

    temp = tf.concat([temp_L, temp_H], 2)

    temp = tf.reshape(temp, [x_n, x_h, 2 * x_w, x_d, x_c])

    recon = tf.transpose(temp, [0, 2, 1, 3, 4])

    return recon


def reconstruct_3D(LLL, HLL, LHL, HHL, LLH, HLH, LHH, HHH):
    LLL = tf.transpose(LLL, [0, 3, 2, 1, 4])
    HLL = tf.transpose(HLL, [0, 3, 2, 1, 4])
    LLL, HLL = lifting97_3D_learned_f.lifting97_inverse4(LLL, HLL)
    LL = reconstruct_fun_depth(LLL, HLL)
    LL = tf.transpose(LL, [0, 3, 2, 1, 4])

    LHL = tf.transpose(LHL, [0, 3, 2, 1, 4])
    HHL = tf.transpose(HHL, [0, 3, 2, 1, 4])
    LHL, HHL = lifting97_3D_learned_f.lifting97_inverse5(LHL, HHL)
    HL = reconstruct_fun_depth(LHL, HHL)
    HL = tf.transpose(HL, [0, 3, 2, 1, 4])

    LLH = tf.transpose(LLH, [0, 3, 2, 1, 4])
    HLH = tf.transpose(HLH, [0, 3, 2, 1, 4])
    LLH, HLH = lifting97_3D_learned_f.lifting97_inverse6(LLH, HLH)
    LH = reconstruct_fun_depth(LLH, HLH)
    LH = tf.transpose(LH, [0, 3, 2, 1, 4])

    LHH = tf.transpose(LHH, [0, 3, 2, 1, 4])
    HHH = tf.transpose(HHH, [0, 3, 2, 1, 4])
    LHH, HHH = lifting97_3D_learned_f.lifting97_inverse7(LHH, HHH)
    HH = reconstruct_fun_depth(LHH, HHH)
    HH = tf.transpose(HH, [0, 3, 2, 1, 4])

    LL = tf.transpose(LL, [0, 2, 1, 3, 4])
    HL = tf.transpose(HL, [0, 2, 1, 3, 4])

    LL, HL = lifting97_3D_learned_f.lifting97_inverse2(LL, HL)

    L = reconstruct_fun(LL, HL)
    L = tf.transpose(L, [0, 2, 1, 3, 4])

    LH = tf.transpose(LH, [0, 2, 1, 3, 4])
    HH = tf.transpose(HH, [0, 2, 1, 3, 4])

    LH, HH = lifting97_3D_learned_f.lifting97_inverse3(LH, HH)

    H = reconstruct_fun(LH, HH)
    H = tf.transpose(H, [0, 2, 1, 3, 4])

    L, H = lifting97_3D_learned_f.lifting97_inverse1(L, H)

    recon = reconstruct_fun(L, H)

    return recon


def concate_volume(LLL, LLH, HLL, HLH, LHL, LHH, HHL, HHH):
    up_L = tf.concat([LLL, HLL], 2)
    bot_L = tf.concat([LHL, HHL], 2)
    up_H = tf.concat([LLH, HLH], 2)
    bot_H = tf.concat([LHH, HHH], 2)
    up = tf.concat([up_L, up_H], 3)
    bot = tf.concat([bot_L, bot_H], 3)
    volume = tf.concat([up, bot], 1)
    return volume
