# Deep Image Homography Estimation  
###### 本篇论文是基于监督学习的神经网络，网络以两幅合成的patch作为输入（两者间存在已知的单应变换关系H），预测输出对应H的8个参数（四个顶点的偏移量）。因为数据集是手工合成的，因而该网络的ground truth也是已知的，所以该思路是在监督下训练神经网络。<br/><br/>


> - 论文来源：[DeTone, D., Malisiewicz, T., & Rabinovich, A. (2016). Deep image homography estimation.(pdf)](https://arxiv.org/pdf/1606.03798)
> - 数据集：合成数据集MSCOCO2014/2017
> - 参考主页（源码未开源）：
>> 1. [alexhagiopol](https://github.com/alexhagiopol/deep_homography_estimation)
>> 2. [mazenmel](https://github.com/mazenmel/Deep-homography-estimation-Pytorch)
>> 3. [**mez**](https://github.com/mazenmel/Deep-homography-estimation-Pytorch)(包含数据集预处理可视化全过程的ipynb文件)
