from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from math import log
import numpy as np
import os
import wavelet_3D_learned_f
import scale_quant
import entropy_codec
import creat_Long_context
import SimpleITK as sitk
import EDEH

decomposition_step = 2


def load_nii(path):
    print(path.split("/")[-1], "loaded!")
    nii = sitk.ReadImage(path)
    nii = sitk.GetArrayFromImage(nii)
    return nii


def save_nii(img, path):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, path)
    print(path.split("/")[-1], "saving succeed!")


def graph(x):
    LLL = x

    HLL_collection = []
    LHL_collection = []
    HHL_collection = []
    LLH_collection = []
    HLH_collection = []
    LHH_collection = []
    HHH_collection = []

    # forward transform, bior4.4

    with tf.variable_scope('wavelet', reuse=tf.AUTO_REUSE):
        for i in range(decomposition_step):
            LLL, HLL, LHL, HHL, LLH, HLH, LHH, HHH = wavelet_3D_learned_f.decomposition(LLL)

            HLL_collection.append(HLL)
            LHL_collection.append(LHL)
            HHL_collection.append(HHL)
            LLH_collection.append(LLH)
            HLH_collection.append(HLH)
            LHH_collection.append(LHH)
            HHH_collection.append(HHH)


    with tf.variable_scope('quant', reuse=tf.AUTO_REUSE):

        static_QP = tf.get_variable('static_QP', initializer=1 / 1)


    LLL, HLL_collection, LHL_collection, HHL_collection, LLH_collection, HLH_collection, LHH_collection, HHH_collection = scale_quant.quant(
        LLL,
        HLL_collection,
        LHL_collection,
        HHL_collection,
        LLH_collection,
        HLH_collection,
        LHH_collection,
        HHH_collection,
        static_QP)

    with tf.variable_scope('long_context', reuse=tf.AUTO_REUSE):
        c_HLL, c_LHL, c_HHL, c_LLH, c_HLH, c_LHH, c_HHH = creat_Long_context.context_all(LLL,
                                                                                         HLL_collection,
                                                                                         LHL_collection,
                                                                                         HHL_collection,
                                                                                         LLH_collection,
                                                                                         HLH_collection,
                                                                                         LHH_collection,
                                                                                         HHH_collection,
                                                                                         static_QP=static_QP * 220,
                                                                                         bit_map=1)

    with tf.variable_scope('ce_loss', reuse=tf.AUTO_REUSE):

        ce_loss = entropy_codec.codec(LLL,
                                      HLL_collection,
                                      LHL_collection,
                                      HHL_collection,
                                      LLH_collection,
                                      HLH_collection,
                                      LHH_collection,
                                      HHH_collection,
                                      c_HLL, c_LHL, c_HHL, c_LLH, c_HLH, c_LHH, c_HHH,
                                      static_QP=static_QP * 220)

    LLL, HLL_collection, LHL_collection, HHL_collection, LLH_collection, HLH_collection, \
    LHH_collection, HHH_collection = scale_quant.de_quant(LLL,
                                                          HLL_collection,
                                                          LHL_collection,
                                                          HHL_collection,
                                                          LLH_collection,
                                                          HLH_collection,
                                                          LHH_collection,
                                                          HHH_collection,
                                                          static_QP)

    with tf.variable_scope('wavelet', reuse=tf.AUTO_REUSE):
        for j in range(decomposition_step):
            i = decomposition_step - 1 - j

            LLL = wavelet_3D_learned_f.reconstruct_3D(LLL,
                                                      HLL_collection[i],
                                                      LHL_collection[i],
                                                      HHL_collection[i],
                                                      LLH_collection[i],
                                                      HLH_collection[i],
                                                      LHH_collection[i],
                                                      HHH_collection[i])

    LLL = LLL / 255. - 0.5

    with tf.variable_scope('post_process', reuse=tf.AUTO_REUSE):
        LLL = EDEH.EDEH(LLL)

    LLL = (LLL + 0.5) * 255.

    d_loss = tf.losses.mean_squared_error(x, LLL)

    loss = d_loss + ce_loss * 4.

    return loss, ce_loss, d_loss, LLL


h_in = 64
w_in = 64
z_in = 64

x = tf.placeholder(tf.float32, [1, z_in, h_in, w_in, 1])

loss, ce_loss, d_loss, LLL = graph(x)

saver = tf.train.Saver()

rate_all = []
psnr_all = []

with tf.Session() as sess:
    saver.restore(sess, '/data/xiaoban/EM_image_small_raw/finetune_attention4/my-model-600')

    for batch_index in range(1):

        path1 = '/data/xiaoban/EM_image_small_raw/attention_test'
        f1 = os.listdir(path1)
        for k in range(0, len(f1)):

            print('img_ID:', k + 1)

            tif = load_nii(os.path.join('/data/xiaoban/EM_image_small_raw/attention_test', f1[k]))

            tif = np.reshape(tif, (1, z_in, h_in, w_in, 1))

            img = np.asarray(tif, dtype=np.float32)

            rate, d_eval, recon_eval = sess.run([ce_loss, d_loss, LLL],
                                                feed_dict={x: img})

            psnr_ = 10 * log(65535. * 65535. / d_eval, 10)

            print('PSNR:', psnr_)
            print('rate:', rate)

            rate_all.append(rate)
            psnr_all.append(psnr_)

            f_txt = open('/output/ct.txt', 'a')
            f_txt.write('ites = %d, rate = %.6f' % (k, rate))
            f_txt.write('ites = %d, psnr = %.6f' % (k, psnr_))
            f_txt.write('\r\n')
            f_txt.close()

            if not os.path.exists('/output/ct/'):
                os.mkdir('/output/ct/')
            if not os.path.exists('/output/recon'):
                os.mkdir('/output/recon')
            recon_eval = recon_eval[0, :, :, :, 0]

            save_nii(recon_eval, '/output/recon/%5s.nii.gz' % str(k).zfill(5))
        rate_all = np.array(rate_all)
        psnr_all = np.array(psnr_all)
        print('rate_mean:', np.mean(rate_all))
        print('psnr_mean:', np.mean(psnr_all))

        f_txt = open('/output/ct.txt', 'a')
        f_txt.write('rate = %.6f' % (np.mean(rate_all)))
        f_txt.write('psnr = %.6f' % (np.mean(psnr_all)))
        f_txt.write('\r\n')
        f_txt.close()

    sess.close()
