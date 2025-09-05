import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits, load_iris
import spectral
import torch
import math
import scipy as sp
from scipy.io import loadmat
import spectral as spy
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tensor_function import Patch, getU, kmode_product

# image = loadmat('./Indian_pines.mat')['indian_pines_corrected']
# label = loadmat('./Indian_pines_gt.mat')['indian_pines_gt']
# image=loadmat('/data/LOH/Tensor_RPCA-master/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
# label=loadmat('/data/LOH/Tensor_RPCA-master/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
image = loadmat('/data/LOH/Tensor_RPCA-master/dataset/Salinas/Salinas.mat')['salinas_corrected']
label = loadmat('/data/LOH/Tensor_RPCA-master/dataset/Salinas/Salinas.mat')['salinas_gt']
PATCH_SIZE = 9
t_all = [1000]
num_runs = 1

for t in t_all:
    for run in range(num_runs):
        '''Divide the training set and test set and randomly select 10% as the training set'''
        [Height, Width, Band] = image.shape
        data = image.astype(float)
        for band in range(Band):
            data[:, :, band] = (data[:, :, band] - np.min(data[:, :, band])) / (
                    np.max(data[:, :, band]) - np.min(data[:, :, band]))
        # noise
        np.random.seed(42)
        noisy_image = np.copy(image).astype(np.float64)
        selected_bands = np.random.choice(image.shape[2], size=30, replace=False)
        for i in selected_bands:
            variance = np.random.uniform(0, 0.5)
            noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
            noisy_image[..., i] += noise
            # 随机生成胡椒和盐噪声的比例(0,0.3)
            salt = np.random.rand(*image[..., i].shape) < 0.05  # 盐噪声（值为255）
            pepper = np.random.rand(*image[..., i].shape) < 0.05  # 胡椒噪声（值为0）
            noisy_image[salt, i] = np.max(image[..., i])  # 设置盐噪声为max
            noisy_image[pepper, i] = np.min(image[..., i])  # 设置胡椒噪声为min
        image = noisy_image

        image_pad = np.zeros((Height + PATCH_SIZE - 1, Width + PATCH_SIZE - 1, Band))
        for band in range(Band):
            # image_pad[:, :, band] = np.pad(data[:, :, band], int((PATCH_SIZE - 1) / 2), 'symmetric')
            mean_value = np.mean(data[:, :, band])
            image_pad[:, :, band] = np.pad(data[:, :, band], int((PATCH_SIZE - 1) / 2), mode='constant',constant_values=mean_value)
        data_patch_list = []
        label_patch_list = []
        for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] + int((PATCH_SIZE - 1) / 2)):
            for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] + int((PATCH_SIZE - 1) / 2)):
                if label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)] != 0:
                    cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2),
                                      PATCH_SIZE)
                    data_patch_list.append(torch.from_numpy(cut_patch.transpose(1, 2, 0)))
                    label_patch_list.append(label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)])

        # random_idx = np.load('/data/LOH/Tensor_RPCA-master/random_idx/random_idx.npy')
        random_idx_dict = np.load('/data/LOH/MTensorLPP-main/random_idx_salinas_1%.npy', allow_pickle=True).item()
        random_idx = []
        for class_label, indices in random_idx_dict.items():
            random_idx.extend(indices['train_indices'])

        X_train, train_label, X_test, test_label = [], [], [], []
        for m in random_idx:
            X_train.append(data_patch_list[m])
            train_label.append(label_patch_list[m])
        idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
        for m in idx_test:
            X_test.append(data_patch_list[m])
            test_label.append(label_patch_list[m])

        newshape_set = [[9,9,30],[9, 9, 2],[9, 9, 5],[9, 9, 10],[9, 9, 15],[9, 9, 20],[9, 9, 25],[9, 9, 30],[9, 9, 35],[9, 9, 40],[9, 9, 45],[9, 9, 50]]
        # newshape_set = [9, 9, 30]
        k_near = 10
        PATCH_SIZE = 9
        for newshape in newshape_set:
            all_class_accs = []
            all_aa = []
            all_oa = []
            all_kappa = []

            U1, U2, U3 = getU(newshape, k_near, X_train, PATCH_SIZE, Band, t)
            l = len(X_train)
            x_train = []
            for i in range(l):
                x = kmode_product(X_train[i], U1, 1)
                x = kmode_product(x, U2, 2)
                x = kmode_product(x, U3, 3)
                x_train.append(x.reshape(newshape[0] * newshape[1] * newshape[2]))
            x_train_np = np.array([item.detach().numpy() for item in x_train])
            x_train = torch.tensor(x_train_np)

            l_t = len(X_test)
            x_test = []
            for i in range(l_t):
                x = kmode_product(X_test[i], U1, 1)
                x = kmode_product(x, U2, 2)
                x = kmode_product(x, U3, 3)
                x_test.append(x.reshape(newshape[0] * newshape[1] * newshape[2]))
            x_test_np = np.array([item.detach().numpy() for item in x_test])
            x_test = torch.tensor(x_test_np)

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(x_train, train_label)
            y_pred = knn.predict(x_test)

            # 计算每一类精度
            unique_classes = np.unique(test_label)
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

            # 最后一次运行时生成分类图
            if run == num_runs - 1:
                all_data_reduced = []
                # 直接使用 data_patch_list 进行降维
                for i in range(len(data_patch_list)):
                    x = kmode_product(data_patch_list[i], U1, 1)
                    x = kmode_product(x, U2, 2)
                    x = kmode_product(x, U3, 3)
                    all_data_reduced.append(x.reshape(newshape[0] * newshape[1] * newshape[2]))
                all_data_reduced_np = np.array([item.detach().numpy() for item in all_data_reduced])
                all_data_reduced = torch.tensor(all_data_reduced_np)

                all_pred = knn.predict(all_data_reduced)

                # 重构分类标签矩阵
                predicted_label = np.zeros_like(label)
                rows, cols = np.nonzero(label != 0)
                for i, (row, col) in enumerate(zip(rows, cols)):
                    predicted_label[row, col] = all_pred[i]

                # 计算 OA 精度并格式化
                oa_mean = np.mean(all_oa)
                oa_str = f"{oa_mean * 100:.2f}"  # 保留两位小数

                # 修改图片标题，添加 OA 精度
                # title = f"TLPP-longkou OA: {oa_str}%"
                # view1 = spy.imshow(classes=predicted_label, title=title, cmap='viridis')
                # ax = plt.gca()
                # ax.axis('off')
                # plt.savefig(f'TLPP-longkou.jpg', dpi=300, bbox_inches='tight')
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
            print(newshape)
            print(f"OA（总体精度）: {oa_mean * 100:.4f} ± {oa_std * 100:.4f}%")
            print(f"AA（平均精度）: {aa_mean * 100:.4f} ± {aa_std * 100:.4f}%")
            print(f"Kappa: {kappa_mean * 100:.4f} ± {kappa_std * 100:.4f}%")
            for i, c in enumerate(unique_classes):
                print(f"类别 {int(c)} 精度: {class_mean_accs[i] * 100:.4f} ± {class_std_accs[i] * 100:.4f}%")
            print("-" * 50)
