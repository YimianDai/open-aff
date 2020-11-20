# [Attentional Feature Fusion](https://arxiv.org/abs/2009.14082)

## Advantage
- Feature fusion based on attention mechanism
- The unified method of feature fusion, the following are applicable
    > (a)Same Layer  (b)Short Skip    (c)Long Skip

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/app.png" width="420px"  height="380px" alt="" >
</div>


## Description
- Modified based on Pytorch official ResNet.
- Support AFFResNet,AFFResNeXt,etc.
- Support AFF,iAFF fusion operation.

## Use

### AFF/iAFF
```python
from fusion import AFF, iAFF
# x,residual  [B,C,H,W]
fusion_mode = AFF(channels=C)
x = fusion_mode(x, residual)
```

### NetWork
- resnet 18/34/50/101/152
- resnext50_32x4d / resnext101_32x8d
- wide_resnet50_2 / wide_resnet101_2


| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `fuse_type` (str,default: DAF) | support AFF,iAFF,DAF |
| `small_input` (bool,default: False) | WH<=112:True |


```python
import resnet50
net = resnet50(fuse_type='DAF',small_input=False)
pred = net(imgs)
```

## Framework
![](https://github.com/bobo0810/imageRepo/blob/master/img/AFF.png)

## Reference
 [MXNet Version](https://github.com/YimianDai/open-aff)











