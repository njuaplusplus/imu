# Official Code for ImU (IEEE S&P 2023)

This is the PyTorch implementation for IEEE S&P 2023 paper "ImU: Physical Impersonating Attack for Face Recognition System with Natural Style Changes". This implementation is heavily based on our implementation of our NDSS 2022 paper [MIRROR](https://github.com/njuaplusplus/mirror).

## Environment

```
conda env create -f environment.yml
conda activate imu
```

## Running examples

### 1. Download cache files

```
python my_download_cache.py
```

### 2. White-box attack

Attack ResNet50 and test results.

```
zsh scripts/run_vggface2_resnet50_whitebox.sh
zsh scripts/run_vggface2_resnet50_whitebox_test.sh
```

Attack IRSE50 and test results.

```
zsh scripts/run_fr_irse50.sh
zsh scripts/run_fr_irse50_test.sh
```

### 3. Black-box attack

Attack ResNet50 and test results.

```
zsh scripts/run_vggface2_resnet50_blackbox.sh
zsh scripts/run_vggface2_resnet50_blackbox_test.sh
```

Attack IRSE50 and test results.

```
zsh scripts/run_fr_irse50_ensemble.sh
zsh scripts/run_fr_irse50_ensemble_test.sh
```

## Embedding with auto-alignment

```
python my_embed_with_auto_alignment.py ./my_examples/me2.jpg
```

## Acknowledgement

The StyleGAN models are based on [genforce/genforce](https://github.com/genforce/genforce).

VGG16/VGG16BN/Resnet50 models are from [their official websites](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html).

InceptionResnetV1 is from [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch).

SphereFace is from [clcarwin/sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch).

Face verification models in `Pretrained_FR_Models` folder are from [TencentYoutuResearch/Adv-Makeup](https://github.com/TencentYoutuResearch/Adv-Makeup#pre-trained-models).


## BibTex
Please cite our work as follows for any purpose of usage.

```
@inproceedings{An.ImU.SP.2023,
    title={ImU: Physical Impersonating Attack for Face Recognition System with Natural Style Changes},
    author={An, Shengwei and Yao, Yuan and Xu, Qiuling and Ma, Shiqing and Tao, Guanhong and Cheng, Siyuan and Zhang, Kaiyuan and Liu, Yingqi and Shen, Guangyu and Kelk, Ian and Zhang, Xiangyu},
    booktitle={IEEE Symposium on Security and Privacy (SP 2023)},
    year={2023}
}
```
