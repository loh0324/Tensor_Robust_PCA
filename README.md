# Tensor Robust PCA with MTLPP and TLPP

该项目实现了基于张量鲁棒主成分分析（Tensor Robust PCA, TRPCA）结合多线性局部保留投影（MTLPP）和张量局部保留投影（TLPP）的高光谱图像处理与分类方法。通过对高光谱图像进行去噪、特征提取和降维，最终使用KNN分类器实现地物分类。

## 项目结构

```
Tensor_Robust_PCA/
├── tensor_function.py       # 核心张量操作函数（包含MTLPP、TLPP相关实现）
├── MTLPP.py                # 基于MTLPP的高光谱图像分类实现
├── reproduction_TLPP.py    # 基于TLPP的高光谱图像分类实现
├── TRPCA_MTLPP.py          # TRPCA去噪+MTLPP的分类实现
├── TRPCA_LPP.py            # TRPCA去噪+PCA的分类实现
└── .idea/                  # 项目配置文件
```

## 依赖环境

- Python 3.x
- NumPy
- PyTorch
- SciPy
- scikit-learn
- spectral (用于高光谱图像处理与可视化)
- matplotlib (用于结果可视化)

## 主要功能

1. **高光谱图像去噪**：基于TRPCA（张量鲁棒主成分分析）实现高光谱图像的去噪处理，分离低秩分量（干净图像）和稀疏分量（噪声）
   
2. **特征提取与降维**：
   - MTLPP（多线性局部保留投影）：一种张量数据的降维方法，保留数据的局部结构
   - TLPP（张量局部保留投影）：另一种张量降维方法
   - PCA（主成分分析）：作为对比的传统降维方法

3. **分类任务**：使用KNN（K近邻）分类器对降维后的特征进行分类，评估指标包括：
   - OA（总体精度）
   - AA（平均精度）
   - Kappa系数
   - 各类别精度

## 使用方法

1. **数据准备**：
   - 项目支持多种高光谱数据集（Indian Pines、Salinas、WHU-Hi-LongKou等）
   - 需将数据文件放在对应路径下，可通过修改代码中的`loadmat`路径指定数据集

2. **参数配置**：
   - 可调整参数包括：补丁大小（PATCH_SIZE）、近邻数（k_near）、降维后的维度（newshape）、运行次数（num_runs）等
   - 噪声设置：可配置高斯噪声和椒盐噪声的参数

3. **运行程序**：
   - 运行`MTLPP.py`：使用MTLPP方法进行分类
   - 运行`reproduction_TLPP.py`：使用TLPP方法进行分类
   - 运行`TRPCA_MTLPP.py`：先进行TRPCA去噪，再使用MTLPP进行分类
   - 运行`TRPCA_LPP.py`：先进行TRPCA去噪，再使用PCA进行分类

## 结果输出

- 分类精度指标（OA、AA、Kappa、各类别精度）
- 分类结果可视化图像

## 备注

- 代码中包含了数据预处理（归一化）、训练集/测试集划分、噪声添加等辅助功能
- 可通过修改随机索引文件（random_idx.npy）控制训练样本的选择
- 不同数据集的路径需要根据实际存储位置进行调整
