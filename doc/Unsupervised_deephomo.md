# Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images  
###### 本篇论文提出一个基于非监督学习单应矩阵的神经网络，是在Daniel D.T.等人发表的Deep Homography Estimation论文上的一个改进方案。Daniel的网络在合成数据集上虽然能呈现较好的结果，但在真实数据集上仍然存在较大的误差。本文通过重新定义损失函数，将4-points的差值转化为图像像素级的差值来实现反向传播。论文中提出利用Spatial Transform Layer实现图像的变形。<br/><br/><br/>


> - 论文来源：[Nguyen, T., Chen, S. W., Shivakumar, S. S., Taylor, C. J., & Kumar, V. (2018). Unsupervised deep homography: A fast and robust homography estimation model. IEEE Robotics and Automation Letters, 3(3), 2346-2353..(pdf)](https://arxiv.org/pdf/1709.03966)
> - 数据集：1. 合成数据集MSCOCO2014/2017     2. 真实数据集(未开源)
> - 参考主页：[tynguyen](https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018)


<br/><br/><br/>


## 1. 主要思路

#### 矩阵H形式化
单应性矩阵的表达𝐻_𝑚𝑎𝑡𝑟𝑖𝑥转化为对第一幅图像四个顶点坐标的8个维度（𝑥_𝑖, 𝑦_𝑖, 𝑖=1,2,…8 ）的偏移量𝐻_4𝑝𝑜𝑖𝑛𝑡。
#### 4-points形式化的好处
相比较3x3的参数化形式，由于H中混合了旋转成分、平移成分、尺度成分和错切成分。平移分量比旋转和错切分量在数值上变换更大。当计算矩阵误差时，两者对矩阵值的影响都很大，但旋转分量的差值对L2损失函数所造成的**影响比重**比平移向量小。
<br/>
<div align=center>
<img src="https://github.com/Leeing98/DeepHomographyEstimation/blob/main/img_folder/4points_parameterization.png" width="460" height="200">
</div>


<br/><br/><br/>
## 2. 合成数据集
> - 来源：[MSCOCO](https://cocodataset.org/#download) 2014 train/val/testing
> - 合成方法：
>> 1. 对于MSCOCO数据集的图像，选定一个**随机的位置点p**
>> 2. 以p为patch的左上角顶点，确定长宽均为128的**patchA**
>> 3. 对patchA的四个顶点做x,y轴上随机的摆动，得到4-points长度为**8维的偏移量**
>> 4. 四个顶点变换前的坐标到变换后的坐标存在一个单应变换矩阵HAB，将原图像乘上HBA（逆矩阵）得到warped图像
>> 5. 在warped图像上同一位置p上取一个128x128大小的patch名为**patchB**


<br/>
<div align=center>
  <img src="https://github.com/Leeing98/DeepHomographyEstimation/blob/main/img_folder/Training%20Data%20Generation.png" width="500" height="500">
  </div>
  
### 合成数据集代码示例



<br/><br/><br/>
## 3. 网络结构
  
<br/>


```python

```


<br/><br/>
## 4. 实验结果
实验分为两个网络——回归网络和分类网络。  

- 回归网络的输出为8维张量，直接对应4-points的8个偏移量。GT是8个偏移量
- 分类网络的输出是8\*21大小的张量，每个21维的向量表示在该坐标值在取值范围\[10,-10]的概率。GT由正确的偏移量确定，eg：某点x坐标的偏移为-3，则21维向量里代表-3的那一位概率为1，其余都为0。

<br/>

<center>
<figure>
  <img src = "https://github.com/Leeing98/DeepHomographyEstimation/blob/main/img_folder/Regression%20HomographyNet.png"  width = "400" align = left>
  <img src = "https://github.com/Leeing98/DeepHomographyEstimation/blob/main/img_folder/Classification%20HomographyNet.png"  width = "400" align = right>
</figure>
 </center>


<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
#### 结论


<br/><br/><br/>


## 5.复现实验
### 合成数据集过程


待补充...

