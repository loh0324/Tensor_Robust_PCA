import numpy as np
from matplotlib import pylab as plt
from numpy.linalg import svd
from PIL import Image
import numpy as np
import spectral
import torch
import math
import scipy as sp
from scipy.io import loadmat
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from tensor_function import Patch,getU,kmode_product,train_test_tensor


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
                U, S, V = svd(D[:, :, i], full_matrices = False)
                S = self.SoftShrink(S, tau)
                S = np.diag(S)
                w = np.dot(np.dot(U, S), V)
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis = 2)
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
        max_iters = 100
        lamb = (max(m, n) * l) ** -0.5
        L = np.zeros((m, n, l), float)
        E = np.zeros((m, n, l), float)
        Y = np.zeros((m, n, l), float)
        iters = 0
        while True:
            iters += 1
            # update L(recovered image)
            L_new = self.SVDShrink(X - E + (1/mu) * Y, 1/mu)
            # update E(noise)
            E_new = self.SoftShrink(X - L_new + (1/mu) * Y, lamb/mu)

            Y += mu * (X - L_new - E_new)
            mu = min(rho * mu, mu_max)
            if self.converged(L, E, X, L_new, E_new) or iters >= max_iters:
                return L_new, E_new
            else:
                L, E = L_new, E_new
                print(iters, np.max(X - L - E))


# Load Data
# image = loadmat('./Indian_pines.mat')['indian_pines_corrected']
# label = loadmat('./Indian_pines_gt.mat')['indian_pines_gt']
# image = loadmat('/data/LOH/Tensor_RPCA-master/dataset/PaviaU/PaviaU.mat')['paviaU']
# label = loadmat('/data/LOH/Tensor_RPCA-master/dataset/PaviaU/PaviaU.mat')['paviaU_gt']
# image = loadmat('/data/LOH/MTensorLPP-main/dataset/ZY1-02D盐城数据/ZY_YC_data147.mat')['Data']
# label = loadmat('/data/LOH/MTensorLPP-main/dataset/ZY1-02D盐城数据/ZY_YC_gt7.mat')['DataClass']
image=loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
label=loadmat('/data/LOH/MTensorLPP-main/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
# X = np.array(Image.open(r'photo.jpg'))

'''add noise(make some pixels black at the rate of 10%)'''
Height,Width,Band=image.shape
# image=image.astype(float)
# for band in range(Band):
#     image[:,:,band]=(image[:,:,band]-np.min(image[:,:,band]))/(np.max(image[:,:,band])-np.min(image[:,:,band]))

# np.random.seed(42)
# noisy_image = np.copy(image).astype(np.float64)
# selected_bands = np.random.choice(image.shape[2], size = 60, replace=False)
# for i in selected_bands:
#     variance = np.random.uniform(0, 0.5)
#     noise = np.random.normal(0, np.sqrt(variance), size=image[..., i].shape)
#     noisy_image[..., i] += noise
#     # 随机生成胡椒和盐噪声的比例(0,0.3)
#     salt = np.random.rand(*image[..., i].shape) < 0.05  # 盐噪声（值为255）
#     pepper = np.random.rand(*image[..., i].shape) < 0.05  # 胡椒噪声（值为0）
#     noisy_image[salt, i] = np.max(image[..., i])  # 设置盐噪声为max
#     noisy_image[pepper, i] = np.min(image[..., i])  # 设置胡椒噪声为min
# image = noisy_image

random_idx = np.load('random_idx.npy')
# random_idx_dict = np.load('/data/LOH/Tensor_RPCA-master/random_idx/random_idx_longkou_0.2%.npy', allow_pickle=True).item()
# random_idx = []
# for class_label, indices in random_idx_dict.items():
#     random_idx.extend(indices['train_indices'])

'''image denoising'''
TRPCA = TRPCA()
# L, E = TRPCA.ADMM(image) # L是低秩张量分量，E是稀疏张量噪声分量
# L = np.array(L).astype(np.uint8)
# E = np.array(E).astype(np.uint8)
# np.save('L-longkou',L)
L = np.load('L-longkou-guiyi.npy')
x_train,train_label,x_test,test_label = train_test_tensor(L,label,random_idx)

newshape = [9,9,30]
k_near=10
PATCH_SIZE=9
t = 1000
U1,U2,U3=getU(newshape,k_near,x_train,PATCH_SIZE,Band,t)
l=len(x_train)
X_train=[]
for i in range(l):
    x=kmode_product(x_train[i],U1,1)
    x=kmode_product(x,U2,2)
    x=kmode_product(x,U3,3)
    X_train.append(x.reshape(newshape[0]*newshape[1]*newshape[2]))
X_train= torch.tensor([item.detach().numpy() for item in X_train])
X_test=[]
l_t=len(x_test)
for i in range(l_t):
    x=kmode_product(x_test[i],U1,1)
    x=kmode_product(x,U2,2)
    x=kmode_product(x,U3,3)
    X_test.append(x.reshape(newshape[0]*newshape[1]*newshape[2]))
X_test= torch.tensor([item.detach().numpy() for item in X_test])

# 进行预测
# 假设测试数据样本存储在 test_samples 变量中
#predictions = knn_classifier.predict(test_samples)
#svc=svm.SVC(C=100,gamma=10,probability=True)
#X_test= torch.tensor([item.detach().numpy() for item in X_test])
#X_train= torch.tensor([item.detach().numpy() for item in X_train])
#svc.fit(x_train,train_label)
#acc=svc.score(x_test,test_label)
#print(acc)

def nn(x_train, train_label,x_test, test_label):
    computedClass = []
    D = np.zeros((len(x_test),len(x_train)))
    for i in range(len(x_test)):
        current_block = x_test[i]
        for j in range(len(x_train)):
            neighbor_block = x_train[j]
            w = current_block-neighbor_block
            d = np.linalg.norm(w)
            D[i,j] = d
    id = np.argsort(D, axis=1)
    count = 0

    computedClass.append(np.array(train_label)[id[:, 0]])
    for w in range(len(x_test)):
        if computedClass[0][w]==test_label[w]:
            count = count+1
    recogRate = count / len(x_test)

    return recogRate
#
acc= nn(X_train, train_label,X_test, test_label)
# knn = KNeighborsClassifier(n_neighbors = 10)
# knn.fit(x_train, train_label)
# knn.score(x_train, train_label)
# acc=knn.score(x_test, test_label)
print(f"{acc * 100:.4f}")


