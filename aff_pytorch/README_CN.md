# [Attentional Feature Fusion](https://arxiv.org/abs/2009.14082)

## 说明
- 第三方实现
- 2020.11.21 已并入[官方仓库](https://github.com/YimianDai/open-aff)
- 2020.11.20 支持AFFResNet, AFFResNeXt
- 2020.11.19 新增MS_CAM, AFF, iAFF

## 作用
- 基于注意力机制的特征融合
- 特征融合的统一方式，以下均适用
    > (a)Same Layer  (b)Short Skip    (c)Long Skip

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/app.png" width="420px"  height="380px" alt="" >
</div>

 

## 使用

### 单特征通道加权 MS_CAM
```python
from fusion import MS_CAM
# x[B,C,H,W]  like SE Module
fusion_mode = MS_CAM(channels=C)
x = fusion_mode(x)
```


### 多特征融合 AFF, iAFF
```python
from fusion import AFF, iAFF
# x,residual  [B,C,H,W]
fusion_mode = AFF(channels=C)
x = fusion_mode(x, residual)
```

### 网络
- resnet 18/34/50/101/152
- resnext50_32x4d / resnext101_32x8d
- wide_resnet50_2 / wide_resnet101_2


| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `fuse_type` (str,default: DAF) | 特征融合类型，支持AFF,iAFF,DAF |
| `small_input` (bool,default: False) | WH<=112为True |


```python
import resnet50
net = resnet50(fuse_type='DAF',small_input=False)
pred = net(imgs)
```

## 算法框架
![](https://github.com/bobo0810/imageRepo/blob/master/img/AFF.png)

## 参考
 [官方MXNet版](https://github.com/YimianDai/open-aff)











