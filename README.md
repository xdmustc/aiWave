# aiWave: Volumetric Image Compression with 3-D Trained Affine Wavelet-like Transform


**Official implementation** of the following paper

Dongmei Xue, Haichuan Ma, Li Li, Dong Liu, and Zhiwei Xiong, aiWave: Volumetric Image Compression with 3-D Trained Affine Wavelet-like Transform. Submit to IEEE Transactions on Medical Imaging.

![](https://github.com/xdmustc/aiWave/blob/main/overview.PNG)

## Dependencies

- Python 3.6 
- TensorFlow 1.14.0
- Numpy 1.17.3
- pandas 1.1.5
- dask 2.9.1
- [SimpleITK 2.1.1.2](https://pypi.org/project/SimpleITK/) 


## Usage

### 1. Datasets Preparation

We make lossy and lossless compression experiments on totally seven 3D biomedical datasets. All of the information of the datasets can be seen below.

Original websites of these datasets are given:
- [FAFB](https://temca2data.org/)
- [FIB-25](https://bio-protocol.org/prep657)
- [Spleen-CT](http://medicaldecathlon.com/)
- [Heart-MRI](http://medicaldecathlon.com/)
- [Chaos_CT](https://zenodo.org/record/3431873#.YpgoF-hBybh)
- [Attention](https://www.fil.ion.ucl.ac.uk/spm/data/attention/)
- [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/)

All the data is converted to .tif or .nii.gz format and crop to smaller size such as 64*64*64. Download our processed data from [BaiduYun](https://pan.baidu.com/s/1fjuJmnSrjWQBzVBXjoO_EA) (Access code: 7gtd)

For training, using [make_tfrecords.py](https://github.com/xdmustc/aiWave/blob/main/make_tfrecords.py) to make a TensorFlow dataset.


### 2. Command Line of anchors

Several traditional codecs were used as our anchor, including JP3D, JPEG-2000-Part2, HEVC, HEVC-RExt. We published the command lines of them to facilitate the use and reproduction of our results. Also, the config files mentioned below can be download through [BaiduYun](https://pan.baidu.com/s/1DI-NtvrONx2RHRfpBkuZsg) (Access code: 8j3v)

![](https://github.com/xdmustc/aiWave/blob/main/results.PNG)

1. For **JP3D**, the [OpenJPEG 2.3.1](http://www.openjpeg.org/2019/04/02/OpenJPEG-2.3.1-released) software was adopted with the command line below.

- Encode command line:
```shell
./opj_jp3d_compress.exe -i input.bin -m config.img -o output.jp3d -r 5 -T 3DWT -C 3EB > log_encode.log
```

- Decode command line:
```shell
./opj_jp3d_decompress.exe -i output.jp3d -m config.img -O input.bin -o output.bin > log_decode.log
```


2. For **JPEG-2000-Part2**, the [Kakadu 6.1](https://kakadusoftware.com/) software was adopted with the command line below.

- Encode command line:
```shell
kdu_compress.exe -i input.rawl*64@4096 -o output.jpx -jpx_layers * -jpx_space sLUM Sdims="{64,64}" Clayers=4 -rate 320 Mcomponents=64 Msigned=no Mprecision=8 -cpu 0 Ssigned=no,no,no Sprecision=8,8,8 Mvector_size:I4=64 Mvector_coeffs:I4=2048 Mstage_inputs:I5="{0,63}" Mstage_outputs:I5="{0,63}" Mstage_collections:I5="{64,64}" Mstage_xforms:I5="{DWT,0,4,3,0}" Mnum_stages=1 Mstages=5  > log_encode.log
```

- Decode command line:
```shell
kdu_expand -i output.jpx -o input.tif -raw_components 0 -skip_components 0 -cpu 0 -record log.txt > log_decode.log 
```


3. For **HEVC**, the [HM 16.15](https://vcgit.hhi.fraunhofer.de/jvet/HM/-/tree/HM-16.15) software was adopted with the command line below.

- Encode command line:
```shell
./TAppEncoder.exe -c encoder_randomaccess_main_rext.cfg -c config.cfg -i input.yuv -b input.bin -o output.bin --ECU=1 --CFM=1 --ESD=1 --FramesToBeEncoded=64 --QP=10 > log_encode.log 
```

- Decode command line:
```shell
./TAppDecoder.exe -b input.bin -o input.yuv > log_decode.log 
```


4. For **HEVC-RExt**, the [HM 16.15](https://vcgit.hhi.fraunhofer.de/jvet/HM/-/tree/HM-16.15) software was adopted with the command line below.

- Encode command line:
```shell
./TAppEncoder.exe -c encoder_randomaccess_main_rext.cfg -c config.cfg -i input.yuv -b input.bin -o input.yuv --ECU=1 --CFM=1 --ESD=1 --FramesToBeEncoded=64 --QP=25 > log_encode.log
```

- Decode command line:
```shell
./TAppDecoder.exe -b input.bin -o input.yuv > log_decode.log 
```


### 3. Fully-trained Model

During the training process, additive wavelet-like transform was end-to-end trained firstly. Please refer to [lifting97_3D_learned.py](https://github.com/xdmustc/aiWave/blob/main/lifting97_3D_learned.py) and [wavelet_3D_learned.py](https://github.com/xdmustc/aiWave/blob/main/wavelet_3D_learned.py) for additive wavelet-like transform. Then we fiexed the entropy model to train the affine wavelet-like transform with [lifting97_3D_learned_f.py](https://github.com/xdmustc/aiWave/blob/main/lifting97_3D_learned_f.py) and [wavelet_3D_learned_f.py](https://github.com/xdmustc/aiWave/blob/main/wavelet_3D_learned_f.py). Finally, all the models were trained jointly to finetune the parameters.

We provide the **fully-trained model** of our method. Download the releaed model from [BaiduYun](https://pan.baidu.com/s/11RWv8K3BDWcw_S5O1MyPxA) (Access code: xdv4)



## Citation

If you find this work helpful, please consider citing our paper.

```latex
@article{xue2022aiwave,
  title={aiWave: Volumetric Image Compression with 3-D Trained Affine Wavelet-like Transform},
  author={Xue, Dongmei and Ma, Haichuan and Li, Li and Liu, Dong and Xiong, Zhiwei},
  journal={arXiv preprint arXiv:2203.05822},
  year={2022}
}
```

## Contact

If you have any problem about the released code, please do not hesitate to contact me with email (xdm1@mail.ustc.edu.cn).
