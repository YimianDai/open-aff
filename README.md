Attentional Feature Fusion
==============

MXNet/Gluon code for "Attentional Feature Fusion" <>

What's in this repo so far:

 * Code, trained models, and training logs for CIFAR-10, CIFAR-100, and ImageNet
 
## Requirements
 
Install [MXNet](https://mxnet.apache.org/) and [Gluon-CV](https://gluon-cv.mxnet.io/):
  
```
pip install --upgrade mxnet-cu100 gluoncv
```

## Experiments 

**All trained model params and training logs are in `./params`**

### AFF-ResNet-50 on ImageNet

Training script:
```python
python train_imagenet.py --askc-type ASKCFuse --mixup --mode hybrid --lr 0.075 --lr-mode cosine --num-epochs 180 --batch-size 128 --num-gpus 2 -j 48 --warmup-epochs 5 --dtype float16 --use-rec --last-gamma --no-wd --label-smoothing --save-dir params_resnet50_v1b_best_AFF --logging-file resnet50_v1b_best_AFF.log
```

### iAFF-ResNet-50 on ImageNet

Training script:
```python
python train_imagenet.py --askc-type ResGlobLocaforGlobLocaCha --mixup --mode hybrid --lr 0.075 --lr-mode cosine --num-epochs 180 --batch-size 128 --num-gpus 2 -j 48 --warmup-epochs 5 --dtype float16 --use-rec --last-gamma --no-wd --label-smoothing --save-dir params_resnet50_v1b_best_iAFF --logging-file resnet50_v1b_best_iAFF.log
```

### AFF-ResNeXt-50-32x4d on ImageNet

Training script:
```python
python train_imagenet.py --askc-type ASKCFuse --model resnext50_32x4d_askc --mode hybrid --lr 0.075 --lr-mode cosine --num-epochs 240 --batch-size 128 --num-gpus 2 -j 48 --use-rec --dtype float16 --warmup-epochs 5 --last-gamma --no-wd --label-smoothing --mixup --save-dir params_resnext50_32x4d_aff_best --logging-file resnext50_32x4d_aff_best.log
```

### iAFF-ResNeXt-50-32x4d on ImageNet

Training script:
```python
python train_imagenet.py --askc-type ResGlobLocaforGlobLocaCha --model resnext50_32x4d_askc --mode hybrid --lr 0.075 --lr-mode cosine --num-epochs 240 --batch-size 128 --num-gpus 2 -j 48 --use-rec --dtype float16 --warmup-epochs 5 --last-gamma --no-wd --label-smoothing --mixup --save-dir params_resnext50_32x4d_iaff_best --logging-file resnext50_32x4d_iaff_best.log
```

| Architecture                                    | Params    | top-1 err.  |
| --------                                        | -------   | ----------- |
| ResNet-101 [[1]](#1)                            | 42.5M     | 23.2        |
| Efficient-Channel-Attention-Net-101 [[2]](#2)   | 42.5M     | 21.4        |
| Attention-Augmented-ResNet-101 [[3]](#3)        | 45.4M     | 21.3        |
| SENet-101 [[4]](#4)                             | 49.4M     | 20.9        |
| Gather-Excite-$\theta^{+}$-ResNet-101 [[5]](#5) | 58.4M     | 20.7        |
| Local-Importance-Pooling-ResNet-101 [[6]](#6)   | 42.9M     | 20.7        |
| AFF-ResNet-50 (**ours**)                        | **30.3M** | **20.9**    |
| AFF-ResNeXt-50-32x4d (**ours**)                 | **29.9M** | **20.8**    |
| iAFF-ResNet-50 (**ours**)                       | **35.1M** | **20.4**    |
| iAFF-ResNeXt-50-32x4d (**ours**)                | **34.7M** | **20.2**    |

### Aff-ResNet-32 on CIFAR-100

Training script:
```python
python train_cifar.py --model resnet --dataset cifar100 --blocks 5 --channel-times 4 --gpus 0 --start-layer 1 --num-epochs 400 --mode hybrid -j 2 --batch-size 128 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 300,350
```

### Aff-ResNeXt-38-32x4d on CIFAR-100

Training script:
```python
python train_cifar.py --model resnext38_32x4d --dataset cifar100 --gpus 0,1,2 --num-epochs 640 --mode hybrid -j 28 --batch-size 128 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 300,450
```


| Architecture                                            | Params   | Accuracy    |
| --------                                                | -------  | ----------- |
| Attention-Augmented-Wide-ResNet-28-10 <a id="3">[3]</a> | 36.2M    | 81.6        |
| SENet-29 <a id="4">[4]</a>                              | 35.0M    | 82.2        |
| SKNet-29  <a id="7">[7]</a>                             | 27.7M    | 82.7        |
| PyramidNet-272-alpha-200 <a id="8">[8]</a>              | 26.0M    | 83.6        |
| Neural Architecture Transfer (NAT-M4) <a id="9">[9]</a> | 9.0M     | 88.3        |
| AutoAugment+PyramidNet+ShakeDrop <a id="10">[10]</a>    | 26.0M    | 89.3        |
| AFF-ResNet-32 (**ours**)                                | **5.0M** | **89.3**    |
| AFF-ResNeXt-38-32x4d (**ours**)                         | **7.8M** | **90.3**    |



## References

<a id="1">[1]</a> 
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun:
Deep Residual Learning for Image Recognition. CVPR 2016: 770-778

<a id="2">[2]</a> 
Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu:
ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. CVPR 2020: 11531-11539

<a id="3">[3]</a> 
Irwan Bello, Barret Zoph, Quoc Le, Ashish Vaswani, Jonathon Shlens:
Attention Augmented Convolutional Networks. ICCV 2019: 3285-3294

<a id="4">[4]</a> 
Jie Hu, Li Shen, Gang Sun:
Squeeze-and-Excitation Networks. CVPR 2018: 7132-7141

<a id="5">[5]</a> 
Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Andrea Vedaldi:
Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks. NeurIPS 2018: 9423-9433

<a id="6">[6]</a> 
Ziteng Gao, Limin Wang, Gangshan Wu:
LIP: Local Importance-Based Pooling. ICCV 2019: 3354-3363

<a id="7">[7]</a> 
Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang:
Selective Kernel Networks. CVPR 2019: 510-519

<a id="8">[8]</a> 
Dongyoon Han, Jiwhan Kim, Junmo Kim:
Deep Pyramidal Residual Networks. CVPR 2017: 6307-6315

<a id="9">[9]</a> 
Zhichao Lu, Gautam Sreekumar, Erik D. Goodman, Wolfgang Banzhaf, Kalyanmoy Deb, Vishnu Naresh Boddeti:
Neural Architecture Transfer. CoRR abs/2005.05859 (2020)

<a id="10">[10]</a> 
Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le:
AutoAugment: Learning Augmentation Strategies From Data. CVPR 2019: 113-123


