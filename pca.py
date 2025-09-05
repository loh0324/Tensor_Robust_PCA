import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split

# 加载数据
image = loadmat('/data/LOH/Tensor_RPCA-master/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
label = loadmat('/data/LOH/Tensor_RPCA-master/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']

# 加载训练样本索引
random_idx_dict = np.load('/data/LOH/MTensorLPP-main/random_idx_longkou_0.2%.npy', allow_pickle=True).item()
random_idx = []
for class_label, indices in random_idx_dict.items():
    random_idx.extend(indices['train_indices'])

# 准备训练和测试数据
# 将图像数据转换为二维数组
rows, cols, bands = image.shape
image_2d = image.reshape(-1, bands)
label_2d = label.flatten()

# 根据索引获取训练数据
x_train = image_2d[random_idx]
train_label = label_2d[random_idx]

# 获取测试数据
test_mask = np.ones(label_2d.shape, dtype=bool)
test_mask[random_idx] = False
x_test = image_2d[test_mask]
test_label = label_2d[test_mask]

# 使用 PCA 进行降维，保留 30 个维度
pca = PCA(n_components=30)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# 创建 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=1)

# 在降维后的数据上训练模型
knn.fit(x_train_pca, train_label)

# 计算训练集准确率
train_acc = knn.score(x_train_pca, train_label)
print(f"训练集准确率: {train_acc:.4f}")

# 计算测试集准确率
acc = knn.score(x_test_pca, test_label)
print(f"测试集准确率: {acc:.4f}")

# 进行预测
y_pred = knn.predict(x_test_pca)

# 计算平均精度（AA）
aa = precision_score(test_label, y_pred, average='macro')
print(f"AA: {aa:.4f}")

# 计算 Kappa 系数
kappa = cohen_kappa_score(test_label, y_pred)
print(f"Kappa: {kappa:.4f}")

# 打印分类报告
print(classification_report(test_label, y_pred))
