Attentional Feature Fusion
==============

MXNet/Gluon code for "Attentional Feature Fusion" <https://arxiv.org/abs/2009.14082>

What's in this repo so far:

 * Code, trained models, and training logs for CIFAR-10, CIFAR-100, and ImageNet
 
## Requirements
 
Install [MXNet](https://mxnet.apache.org/) and [Gluon-CV](https://gluon-cv.mxnet.io/):
  
```
pip install --upgrade mxnet-cu100 gluoncv
```

## Experiments 

All trained model params and training logs are in `./params`

The training commands / shell scripts are in `cmd_scripts.txt`

### CIFAR-100

| Architecture                                    | Params   | Accuracy    |
| --------                                        | -------  | ----------- |
| Attention-Augmented-Wide-ResNet-28-10 [[3]](#3) | 36.2M    | 81.6        |
| SENet-29 [[4]](#4)                              | 35.0M    | 82.2        |
| SKNet-29  [[7]](#7)                             | 27.7M    | 82.7        |
| PyramidNet-272-alpha-200 [[8]](#8)              | 26.0M    | 83.6        |
| Neural Architecture Transfer (NAT-M4) [[9]](#9) | 9.0M     | 88.3        |
| AutoAugment+PyramidNet+ShakeDrop [[10]](#10)    | 26.0M    | 89.3        |
| **AFF-ResNet-32 (ours)**                        | **5.0M** | **89.3**    |
| **AFF-ResNeXt-38-32x4d (ours)**                 | **7.8M** | **90.3**    |

###  ImageNet

| Architecture                                    | Params    | top-1 err.  |
| --------                                        | -------   | ----------- |
| ResNet-101 [[1]](#1)                            | 42.5M     | 23.2        |
| Efficient-Channel-Attention-Net-101 [[2]](#2)   | 42.5M     | 21.4        |
| Attention-Augmented-ResNet-101 [[3]](#3)        | 45.4M     | 21.3        |
| SENet-101 [[4]](#4)                             | 49.4M     | 20.9        |
| Gather-Excite-$\theta^{+}$-ResNet-101 [[5]](#5) | 58.4M     | 20.7        |
| Local-Importance-Pooling-ResNet-101 [[6]](#6)   | 42.9M     | 20.7        |
| **AFF-ResNet-50 (ours)**                        | **30.3M** | **20.9**    |
| **AFF-ResNeXt-50-32x4d (ours)**                 | **29.9M** | **20.8**    |
| **iAFF-ResNet-50 (ours)**                       | **35.1M** | **20.4**    |
| **iAFF-ResNeXt-50-32x4d (ours)**                | **34.7M** | **20.2**    |

<img src=https://raw.githubusercontent.com/YimianDai/imgbed/master/github/aff/Localization_Reduced.jpg width=100%>
<img src=https://raw.githubusercontent.com/YimianDai/imgbed/master/github/aff/SmallObject_Reduced.jpg width=100%>



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


