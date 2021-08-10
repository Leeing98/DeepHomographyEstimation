# Deep Image Homography Estimation  
###### 本篇论文是基于监督学习的神经网络，网络以两幅合成的patch作为输入（两者间存在已知的单应变换关系H），预测输出对应H的8个参数（四个顶点的偏移量）。因为数据集是手工合成的，因而该网络的ground truth也是已知的，所以该思路是在监督下训练神经网络。<br/><br/><br/>


> - 论文来源：[DeTone, D., Malisiewicz, T., & Rabinovich, A. (2016). Deep image homography estimation.(pdf)](https://arxiv.org/pdf/1606.03798)
> - 数据集：合成数据集MSCOCO2014/2017
> - 参考主页（源码未开源）：
>> 1. [alexhagiopol](https://github.com/alexhagiopol/deep_homography_estimation)
>> 2. [mazenmel](https://github.com/mazenmel/Deep-homography-estimation-Pytorch)
>> 3. [**mez**](https://github.com/mazenmel/Deep-homography-estimation-Pytorch)(包含数据集预处理可视化全过程的ipynb文件)

<br/><br/><br/>


## 1. 主要思路
本文希望通过输入两幅大小一致的图像由网络学习得到**8个参数**，对应两幅图像之间存在的单应关系（矩阵H为8DoF）。
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



<br/><br/><br/>
## 3. 网络结构




