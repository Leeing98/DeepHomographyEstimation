# Content-Aware Unsupervised Deep Homography Estimation
###### 本篇论文是基于非监督学习的神经网络，吸收了Daniel监督学习的思路采用深度神经网络估计单应矩阵，预测的形式依然采用四个点的8维偏移量。但本篇区别于前两篇的改进点在于它提出了利用特征图而不是原图像来计算损失函数，并且它提出了mask的思路来实现对图像中纹理较少和运动等影响匹配的情形进行削弱，对图像中特征明显的部分加大权重，以此同时实现类ransac和类Attention机制的作用。<br/><br/><br/>


> - 论文来源：[Content-Aware Unsupervised Deep Homography Estimation.(pdf)](https://arxiv.org/pdf/1909.05983)
> - 数据集：合成数据集MSCOCO2014/2017、视频数据集Content-Aware-DeepH-Data
> - 参考主页：[JirongZhang(项目源码)](https://github.com/JirongZhang/DeepHomography)

<br/><br/><br/>


## 1. 主要思路
本文希望通过输入两幅大小一致的图像由网络学习得到**8个参数**，对应两幅图像之间存在的单应关系（矩阵H为8DoF）。
#### Feature extractor(特征图提取)

#### Mask predictor(mask预测)

#### Homography estimator(单应性估计)




<br/><br/><br/>
## 2. 数据集




<br/><br/><br/>
## 3. 网络结构
如下图所示是本文网络的整体框架，主要集中在上一章节的三个大改进上。输入阶段是两张大小一样的图像
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


