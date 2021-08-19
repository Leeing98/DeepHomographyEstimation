# Content-Aware Unsupervised Deep Homography Estimation
###### 本篇论文是基于非监督学习的神经网络，吸收了Daniel的非监督学习论文和作为。<br/><br/><br/>


> - 论文来源：[Content-Aware Unsupervised Deep Homography Estimation.(pdf)](https://arxiv.org/pdf/1909.05983)
> - 数据集：合成数据集MSCOCO2014/2017、视频数据集Content-Aware-DeepH-Data
> - 参考主页：[JirongZhang(项目源码)](https://github.com/JirongZhang/DeepHomography)

<br/><br/><br/>


## 1. 主要思路
本文希望通过输入两幅大小一致的图像由网络学习得到**8个参数**，对应两幅图像之间存在的单应关系（矩阵H为8DoF）。
#### Feature extractor(特征图提取)

#### Mask predictor(mask预测)

#### Homography estimator(单应性估计)

<br/>
<div align=center>
<img src="../.assets/Content-aware/Homography estimator.png" width="400" >
</div>


<br/><br/><br/>
## 2. 数据集




<br/><br/><br/>
## 3. 网络结构
<br/>
<div align=center>
  <img src="../.assets/Content-aware/Network structure.png" width="700">
  </div>
  
<br/>



```python

```


<br/><br/>
## 4. 实验结果


<br/>




<br/><br/><br/><br/><br/>
#### 结论


<br/><br/><br/>


## 5.复现实验
### 合成数据集过程


