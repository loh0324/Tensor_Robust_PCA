import numpy as np
import torch
from scipy.io import loadmat
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tensor_function import Patch,getU_MTLPP,kmode_product
import spectral
import matplotlib.pyplot as plt
import spectral as spy
from sklearn.metrics import accuracy_score, cohen_kappa_score

image = loadmat('./Indian_pines.mat')['indian_pines_corrected']
label = loadmat('./Indian_pines_gt.mat')['indian_pines_gt']
# image=loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou'].astype(np.float32)
# label=loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
# image=loadmat('/data/LOH/Tensor_RPCA-master/dataset/Salinas/Salinas.mat')['salinas_corrected']
# label=loadmat('/data/LOH/Tensor_RPCA-master/dataset/Salinas/Salinas.mat')['salinas_gt']
# image = loadmat('./PaviaU.mat')['paviaU']
# label = loadmat('./PaviaU.mat')['paviaU_gt']
# image = loadmat('/data/LOH/MTensorLPP-main/dataset/ZY1-02D盐城数据/ZY_YC_data147.mat')['Data']
# label = loadmat('/data/LOH/MTensorLPP-main/dataset/ZY1-02D盐城数据/ZY_YC_gt7.mat')['DataClass']

'''PCA'''
# rows, cols = np.nonzero(label != 0)
# coordinates = np.column_stack((rows, cols))
#
# fea = np.zeros((coordinates.shape[0],image.shape[2]))
# for i,j in enumerate(coordinates):
#     fea[i,:] = image[j[0],j[1],:]
# # fea = loadmat('./Salinas.mat')['fea'] #10249*200
# pca = PCA(n_components=100)  # 降维到30
# pca.fit(fea)
# fea_reduced = pca.transform(fea)
# x = np.zeros((label.shape[0], label.shape[1], 100))
# for i in range(fea.shape[0]):
#     a = coordinates[i][0]
#     b = coordinates[i][1]
#     x[a][b][:] = fea_reduced[i][:]
# image = x.astype(np.float32)

'''add noise(0,0.5)零均值高斯噪声    随机生成胡椒和盐噪声的比例(0,0.3)'''
# image=image.astype(float)
# for band in range(image.shape[2]):
#     image[:,:,band]=(image[:,:,band]-np.min(image[:,:,band]))/(np.max(image[:,:,band])-np.min(image[:,:,band]))
# np.random.seed(42)
# noisy_image = np.copy(image).astype(np.float64)
# selected_bands = np.random.choice(image.shape[2], size = 80, replace=False)
# for i in selected_bands:
#     variance = np.random.uniform(0, 0.5)
#     noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
#     noisy_image[..., i] += noise
#     # 随机生成胡椒和盐噪声的比例(0,0.3)
#     salt = np.random.rand(*image[..., i].shape) < 0.3  # 盐噪声（值为255）
#     pepper = np.random.rand(*image[..., i].shape) < 0.3  # 胡椒噪声（值为0）
#     noisy_image[salt, i] = np.max(image[..., i])  # 设置盐噪声为max
#     noisy_image[pepper, i] = np.min(image[..., i])  # 设置胡椒噪声为min
# image = noisy_image


'''Divide the training set and test set '''
# image=image.astype(float)
# for band in range(image.shape[2]):
#     image[:,:,band]=(image[:,:,band]-np.min(image[:,:,band]))/(np.max(image[:,:,band])-np.min(image[:,:,band]))

data = image
PATCH_SIZE = 9
[Height, Width, Band] = data.shape
image_pad = np.zeros((Height + PATCH_SIZE - 1, Width + PATCH_SIZE - 1, Band))
for band in range(Band):
    mean_value = np.mean(data[:, :, band])
    image_pad[:,:,band]=np.pad(data[:,:,band],int((PATCH_SIZE-1)/2),mode='constant', constant_values=mean_value)
data_patch_list = []
label_patch_list = []
for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] + int((PATCH_SIZE - 1) / 2)):
    for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] + int((PATCH_SIZE - 1) / 2)):
        if label[i-int((PATCH_SIZE - 1) / 2)][j-int((PATCH_SIZE - 1) / 2)] != 0:
            cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2), PATCH_SIZE)  # 没问题
            data_patch_list.append(torch.from_numpy(cut_patch.transpose(1,2,0)))
            label_patch_list.append(label[i-int((PATCH_SIZE - 1) / 2)][j-int((PATCH_SIZE - 1) / 2)])

# random_idx = np.random.choice(len(data_patch_list), int(0.01*len(data_patch_list)), replace=False)
# np.save('random_idx_ZY-HHK-6.npy',random_idx)
random_idx = np.load('random_idx.npy')
# random_idx_dict = np.load('/data/LOH/Tensor_RPCA-master/random_idx_salinas_1%.npy', allow_pickle=True).item()
# random_idx = []
# for class_label, indices in random_idx_dict.items():
#     random_idx.extend(indices['train_indices'])
#
X_train,train_label,X_test,test_label=[],[],[],[]
for m in random_idx:
    X_train.append(data_patch_list[m])
    train_label.append(label_patch_list[m])
idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
for m in idx_test:
    X_test.append(data_patch_list[m])
    test_label.append(label_patch_list[m])

# num_Class = int(max(label.reshape(label.shape[0] * label.shape[1], 1)))
# class_indices_dict = {}
# for class_label in range(1, num_Class + 1):  # 类别标签从1开始
#     # 获取当前类别的样本索引
#     class_indices = [idx for idx, lbl in enumerate(label_patch_list) if lbl == class_label]
#     # 如果样本数量少于50，全放入训练集，否则随机选50个作为训练集
#     # if len(class_indices) <= 10:
#     #     selected_train_indices = class_indices
#     # else:
#     #     selected_train_indices = np.random.choice(class_indices, 20, replace=False)
#     #     selected_test_indices = np.setdiff1d(class_indices, selected_train_indices)
#
#     # 计算该类别的训练集样本数量（百分之3）
#     num_train_samples = max(3, int(len(class_indices) * 0.003))  # 至少选1个样本
#     # 随机选择3%的样本作为训练集
#     selected_train_indices = np.random.choice(class_indices, num_train_samples, replace=False)
#     # 剩余的样本作为测试集
#     selected_test_indices = np.setdiff1d(class_indices, selected_train_indices)
#     # 保存当前类别的训练和测试集索引
#     class_indices_dict[class_label] = {
#         'train_indices': selected_train_indices.tolist(),
#         'test_indices': selected_test_indices.tolist()
#     }
#     # 将选中的样本添加到训练集和测试集中
#     for idx in selected_train_indices:
#         X_train.append(data_patch_list[idx])
#         train_label.append(label_patch_list[idx])
#     for idx in selected_test_indices:
#         X_test.append(data_patch_list[idx])
#         test_label.append(label_patch_list[idx])
#
# np.save("random_idx_salinas_0.3%.npy", class_indices_dict)

newshape=[9,9,30]
k_near=10
t_all=[1000]
num_runs = 1
for t in t_all:
    all_class_accs = []
    all_oa = []
    all_aa = []
    all_kappa = []
    for run in range(num_runs):
        l = len(X_train)
        ci = []
        d = X_train[0].shape[2]
        for i in range(l):
            c_matrix = torch.zeros((d, d))
            xt = X_train[i]
            ui = torch.mean(xt, dim=(0, 1), keepdim=True)
            ui1 = ui.reshape(d, -1)
            for m in range(PATCH_SIZE):
                for n in range(PATCH_SIZE):
                    xt1 = xt[m, n, :].reshape(d, -1)
                    c_matrix = c_matrix + torch.matmul(xt1 - ui1, (xt1 - ui1).T)
            c_matrix = c_matrix / (PATCH_SIZE * PATCH_SIZE - 1)
            ci.append(c_matrix)
        U1,U2,U3=getU_MTLPP(newshape,k_near,X_train,ci,PATCH_SIZE,Band,t)
        l=len(X_train)
        x_train=[]
        for i in range(l):
            x=kmode_product(X_train[i],U1,1)
            x=kmode_product(x,U2,2)
            x=kmode_product(x,U3,3)
            x_train.append(x.reshape(newshape[0]*newshape[1]*newshape[2]))
        # x_train= torch.tensor([item.detach().numpy() for item in x_train])
        x_train_np = np.array([item.detach().numpy() for item in x_train])  # 将列表转为 NumPy 数组
        x_train = torch.tensor(x_train_np)

        x_test=[]
        l_t=len(X_test)
        for i in range(l_t):
            x=kmode_product(X_test[i],U1,1)
            x=kmode_product(x,U2,2)
            x=kmode_product(x,U3,3)
            x_test.append(x.reshape(newshape[0]*newshape[1]*newshape[2]))
        # x_test= torch.tensor([item.detach().numpy() for item in x_test])
        x_test_np = np.array([item.detach().numpy() for item in x_test])  # 将列表转为 NumPy 数组
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
            view1 = spy.imshow(classes=predicted_label, title=f"1-i {oa_str}", cmap='viridis')
            ax = plt.gca()
            ax.axis('off')
            plt.savefig(f'MTLPP-noise-indian.jpg', dpi=300, bbox_inches='tight')
            plt.close()

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
    print(f"t = {t}")
    print(f"OA（总体精度）: {oa_mean * 100:.4f} ± {oa_std * 100:.4f}%")
    print(f"AA（平均精度）: {aa_mean * 100:.4f} ± {aa_std * 100:.4f}%")
    print(f"Kappa: {kappa_mean * 100:.4f} ± {kappa_std * 100:.4f}%")
    for i, c in enumerate(unique_classes):
        print(f"类别 {int(c)} 精度: {class_mean_accs[i] * 100:.4f} ± {class_std_accs[i] * 100:.4f}%")
    print("-" * 50)
