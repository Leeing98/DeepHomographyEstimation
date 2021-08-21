# Content-Aware Unsupervised Deep Homography Estimation
###### 本篇论文是基于非监督学习的神经网络，吸收了Daniel监督学习的思路采用深度神经网络估计单应矩阵，预测的形式依然采用四个点的8维偏移量。但本篇区别于前两篇的改进点在于它提出了利用特征图而不是原图像来计算损失函数，并且它提出了mask的思路来实现对图像中纹理较少和运动等影响匹配的情形进行削弱，对图像中特征明显的部分加大权重，以此同时实现类ransac和类Attention机制的作用。<br/><br/><br/>


> - 论文来源：[Content-Aware Unsupervised Deep Homography Estimation.(pdf)](https://arxiv.org/pdf/1909.05983)
> - 数据集：合成数据集MSCOCO2014/2017、视频数据集Content-Aware-DeepH-Data
> - 参考主页：[JirongZhang(项目源码)](https://github.com/JirongZhang/DeepHomography)

<br/><br/><br/>


## 1. 主要思路
首先根据下图，我们分析基于特征的方法和基于深度学习的方法存在的缺点和不足：
<div align = "center">
<img src = "../.assets/Content-aware/comparison.png" width = "700">  
</div>

在上图中分别取了三种具有代表性的场景：
1. 有运动的前景
2. 包含纹理较少的场景（雪地、海洋或天空等）
3. 光照过强或过弱

图中是两张图叠在一起的结果，其中一张图只叠加了图像的蓝绿通道，另一张图像只保留红色通道。显示红色或青色的地方就是没有对齐的部分，反之显示原色的部分是对齐的。

#### 基于特征的单应估计
> 图中第一列的就是特征估计单应矩阵的结果，出现了很大部分没有对齐的地方，尤其表现在没有纹理和光照过暗的场景下。造成配准结果较差的原因有：
> 1. 匹配点少
> 2. 特征点**分布不均匀**
> 3. 需要过滤场景中**不占主导地位的平面**或**运动的物体**上的特征点


#### 基于深度学习的单应矩阵预测
> 图中第二列为2018年Nguyen团队提出的非监督学习的框架，对真实数据的配准，基于深度学习的方法对于所有情形上的配准结果都比较一致。在低光照情形下，甚至比SIFT的效果好得多，这也印证了Unsupervised论文中的实验结论。

#### 基于内容学习的单应估计
- 本文主要提出了利用一个特征提取网络提取原图像的特征图
- 利用mask预测网络来对图像的内容进行加权，把注意力放在主要信息上来预测单应矩阵，并在计算损失函数时利用mask过滤非主导信息。


<br/><br/><br/>
## 2. 数据集
本文的数据集为Content-Aware-DeepH-Data，是作者团队拍摄的视频数据。视频数据放置在项目源码文件夹内的/Data文件夹内。数据的读取方式是从Train_List.txt文件中读取，每一行记录两帧的文件名。txt的生成方式本文没有给出，是由个人定义的。项目文件中给出的两帧图像间隔的帧数经观察为2~8帧，因此两幅图像之间的重叠基本在70%以上。
<div align="center">
<img src="../.assets/Content-aware/dataset.png" width="600">
</div>

<br/><br/><br/>
## 3. 网络结构
如下图所示是本文网络的整体框架，主要集中在上一章节的三个大改进上。输入阶段是两张大小一样的图像，每幅图像都分别进入一个特征提取网络和一个Mask预测网络。特征提取网络输出的图像$F_a$和mask预测网络输出的图像$M_a$大小一致，最后进行一个矩阵的点乘得到图像$G_a$。两个经过点乘的图像输入单应性估计网络得到预测的8维向量结果。
<br/>
<div align=center>
  <img src="../.assets/Content-aware/Network structure.png" width="700">
  </div>
  
<br/>

以下是本文网络最重要的三个网络：
#### Feature extractor(特征图提取)
<div align=center>
  <img src="../.assets/Content-aware/Feature_extractor.png" width="350">
  </div>
<br/>


#### Mask predictor(mask预测)
<div align=center>
  <img src="../.assets/Content-aware/Mask predictor.png" width="600">
  </div>
<br/>

#### Homography estimator(单应性估计)
<div align=center>
  <img src="../.assets/Content-aware/Homography estimator.png" width="800">
  </div>
<br/>
在此网络中，我们注意到一个问题————该网络的输入图像大小是不固定的。这是因为在单应估计网络这部分，倒数第二层采用了一个全局平均池化层，经过该层的数据变为512维的张量，最后进入全连接层输出一个8维的结果。

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


