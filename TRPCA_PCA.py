import numpy as np
from numpy.linalg import svd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, cohen_kappa_score, classification_report
import spectral as spy


class TRPCA:
    def converged(self, L, E, X, L_new, E_new):
        '''
        judge convered or not
        '''
        eps = 1e-8
        condition1 = np.max(L_new - L) < eps
        condition2 = np.max(E_new - E) < eps
        condition3 = np.max(L_new + E_new - X) < eps
        return condition1 and condition2 and condition3

    def SoftShrink(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = np.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)
        return z

    def SVDShrink(self, X, tau):
        '''
        apply tensor-SVD and soft thresholding
        '''
        W_bar = np.empty((X.shape[0], X.shape[1], 0), complex)
        D = np.fft.fft(X)
        for i in range(X.shape[2]):
            if i < X.shape[2]:
                U, S, V = svd(D[:, :, i], full_matrices=False)
                S = self.SoftShrink(S, tau)
                S = np.diag(S)
                w = np.dot(np.dot(U, S), V)
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis=2)
            if i == X.shape[2]:
                W_bar = np.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))
        return np.fft.ifft(W_bar).real

    def ADMM(self, X):
        '''
        Solve
        min (nuclear_norm(L)+lambda*l1norm(E)), subject to X = L+E
        L,E
        by ADMM
        '''
        m, n, l = X.shape
        rho = 1.1
        mu = 1e-3
        mu_max = 1e10
        max_iters =1000
        lamb = (max(m, n) * l) ** -0.5
        L = np.zeros((m, n, l), float)
        E = np.zeros((m, n, l), float)
        Y = np.zeros((m, n, l), float)
        iters = 0
        while True:
            iters += 1
            # update L(recovered image)
            L_new = self.SVDShrink(X - E + (1 / mu) * Y, 1 / mu)
            # update E(noise)
            E_new = self.SoftShrink(X - L_new + (1 / mu) * Y, lamb / mu)
            Y += mu * (X - L_new - E_new)
            mu = min(rho * mu, mu_max)
            if self.converged(L, E, X, L_new, E_new) or iters >= max_iters:
                return L_new, E_new
            else:
                L, E = L_new, E_new
                print(np.max(X - L - E))


# Load Data
# image = loadmat('./Indian_pines.mat')['indian_pines_corrected']
# label = loadmat('./Indian_pines_gt.mat')['indian_pines_gt']
# image=loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou'].astype(np.float32)
# label=loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
image=loadmat('/data/LOH/Tensor_RPCA-master/dataset/Salinas/Salinas.mat')['salinas_corrected']
label=loadmat('/data/LOH/Tensor_RPCA-master/dataset/Salinas/Salinas.mat')['salinas_gt']


'''add noise(make some pixels black at the rate of 10%)'''
Height,Width,Band=image.shape
image=image.astype(float)
for band in range(Band):
    image[:,:,band]=(image[:,:,band]-np.min(image[:,:,band]))/(np.max(image[:,:,band])-np.min(image[:,:,band]))
#
# np.random.seed(42)
# noisy_image = np.copy(image).astype(np.float64)
# selected_bands = np.random.choice(image.shape[2], size = 60, replace=False)
# for i in selected_bands:
#     variance = np.random.uniform(0, 0.5)
#     noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
#     noisy_image[..., i] += noise
#     # 随机生成胡椒和盐噪声的比例(0,0.3)
#     salt = np.random.rand(*image[..., i].shape) < 0.1  # 盐噪声（值为255）
#     pepper = np.random.rand(*image[..., i].shape) < 0.1  # 胡椒噪声（值为0）
#     noisy_image[salt, i] = np.max(image[..., i])  # 设置盐噪声为max
#     noisy_image[pepper, i] = np.min(image[..., i])  # 设置胡椒噪声为min
# image = noisy_image

# 已过滤0值标签
rows, cols = image.shape[:2]  # 获取原始图像的行数和列数
non_zero_rows, non_zero_cols = np.nonzero(label != 0)
coordinates = np.column_stack((non_zero_rows, non_zero_cols))

# image denoising
TRPCA = TRPCA()
# L, E = TRPCA.ADMM(image) # L是低秩张量分量，E是稀疏张量噪声分量
# np.save('L-indain-guiyi1.npy',L)
L = np.load('L-salinas-guiyi.npy')
data = []
gnd = []
for i in range(coordinates.shape[0]):  # 向量表示的数据
    a = coordinates[i][0]
    b = coordinates[i][1]
    data.append(L[a][b][:])
    gnd.append(label[a][b])
data = np.array(data)
gnd = np.array(gnd)

# 加载训练样本索引
# random_idx = np.load('/data/LOH/Tensor_RPCA-master/random_idx.npy')
random_idx_dict = np.load('/data/LOH/Tensor_RPCA-master/random_idx_salinas_1%.npy', allow_pickle=True).item()
random_idx = []
for class_label, indices in random_idx_dict.items():
    random_idx.extend(indices['train_indices'])
# 根据索引获取训练数据
x_train = data[random_idx]
train_label = gnd[random_idx]

# 获取测试数据
test_mask = np.ones(gnd.shape, dtype=bool)
test_mask[random_idx] = False
x_test = data[test_mask]
test_label = gnd[test_mask]

# 使用 PCA 进行降维，保留 30 个维度
dims = [2,5,10,15,20,25,30,35,40,45,50]
for dim in dims:
    pca = PCA(n_components=dim)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # 创建 KNN 分类器
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train_pca, train_label)
    # knn.fit(x_train, train_label)
    y_pred = knn.predict(x_test_pca)

    all_class_accs = []
    all_aa = []
    all_oa = []
    all_kappa = []

    # 计算每一类精度（过滤掉0）
    unique_classes = np.unique(test_label)
    unique_classes = unique_classes[unique_classes != 0]  # 移除类别0
    class_accs = []
    for c in unique_classes:
        mask = np.array(test_label) == c
        class_acc = accuracy_score(np.array(test_label)[mask], y_pred[mask])
        class_accs.append(class_acc)
    all_class_accs.append(class_accs)

    # 计算 AA
    aa = np.mean(class_accs)
    all_aa.append(aa)

    # 计算 OA
    oa = accuracy_score(test_label, y_pred)
    all_oa.append(oa)

    # 计算 Kappa
    kappa = cohen_kappa_score(test_label, y_pred)
    all_kappa.append(kappa)

    # 重构分类标签矩阵
    all_data_reduced = pca.transform(data)
    all_pred = knn.predict(all_data_reduced)
    # all_data_reduced = data
    # all_pred = knn.predict(all_data_reduced)

    predicted_label = np.zeros_like(label)
    rows, cols = np.nonzero(label != 0)
    for i, (row, col) in enumerate(zip(rows, cols)):
        predicted_label[row, col] = all_pred[i]

    # 计算 OA 精度并格式化
    oa_mean = np.mean(all_oa)
    oa_str = f"{oa_mean * 100:.2f}"  # 保留两位小数

    # 修改图片标题，添加 OA 精度
    # title = f"TRPCA-s{oa_str}"
    # view1 = spy.imshow(classes=predicted_label, title=title, cmap='viridis')
    # ax = plt.gca()
    # ax.axis('off')
    # plt.savefig(f'TRPCA-salinas.jpg', dpi=300, bbox_inches='tight')
    # plt.close()

    # 计算统计结果
    all_class_accs = np.array(all_class_accs)
    class_mean_accs = np.mean(all_class_accs, axis=0)
    class_std_accs = np.std(all_class_accs, axis=0)

    aa_mean = np.mean(all_aa)
    aa_std = np.std(all_aa)

    oa_mean = np.mean(all_oa)
    oa_std = np.std(all_oa)

    kappa_mean = np.mean(all_kappa)
    kappa_std = np.std(all_kappa)

    # 输出统计结果
    print(dim,f"OA（总体精度）: {oa_mean * 100:.4f} ± {oa_std * 100:.4f}%")
    # print(f"AA（平均精度）: {aa_mean * 100:.4f} ± {aa_std * 100:.4f}%")
    # print(f"Kappa: {kappa_mean * 100:.4f} ± {kappa_std * 100:.4f}%")
    # for i, c in enumerate(unique_classes):
    #     print(f"类别 {int(c)} 精度: {class_mean_accs[i] * 100:.4f} ± {class_std_accs[i] * 100:.4f}%")
    print("-" * 50)
