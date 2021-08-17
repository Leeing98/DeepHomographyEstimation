# 数据集说明文档（未完）

## Synthetic数据集
由于本数据集的合成方式最早由Daniel等人在2016年提出，后续的论文皆为该论文上的进一步推广，因此许多后续论文都将该数据集作为对比实验进行测试。后续许多论文以实验证明，该数据集由于合成时仅采用了一个单应矩阵的变换，没有考虑到深度差异，导致训练后的网络在真实数据集上输出结果误差较大。

> 选用本数据集的论文有：
> 1. Deep image Homography Estimation
> 2. Unsupervised Deep Homography: A Fast and Robust Homography Estimation Model
> 3. Content-aware

该合成数据集是用MSCOCO2014的训练集和测试集制作的，具体的合成细节请看来自Daniel团队的论文。<br/>
- 处理前：网站下载的图像分别放在**train2014**、**test2014**、**val2014**三个文件夹下（图像大小不一）
- 处理后：生成新的处理后的文件夹**train2014processed**、**test2014processed**、**val2014processed**。参考的论文以numpy库下的np.save存储为**npy文件**，分别包含两个大小一致的patch和一个GT的矩阵H。

## Content-Aware-DeepH-Data数据集

