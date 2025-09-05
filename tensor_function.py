import numpy as np
import spectral
import torch
import math
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier

def Patch(data,H,W,PATCH_SIZE):
    transpose_array=np.transpose(data,(2,0,1)) #(C,H,W)
    height_slice=slice(H,H+PATCH_SIZE)
    width_slice=slice(W,W+PATCH_SIZE)
    patch=transpose_array[:,height_slice,width_slice]
    return np.array(patch)

def dist(x,k,t):
    n=len(x)
    s=np.zeros((n,n))
    w=np.zeros((n,n))
    d=np.zeros((n,n))
    for i in range(n):
        #log_Ci=log_euclidean(x[i])
        for j in range(i+1,n):
            #log_Cj=log_euclidean(x[j])
            s[i,j]=np.exp(-(math.pow(np.linalg.norm(x[i]-x[j]),2))/(2*t))
    '''knn'''
    s=s+s.T

    for i in range(n):
        s[i,i]=s[i,i]/2
        index_=np.argsort(s[i])[-(k):]
        w[i,index_]=s[i,index_]
        w[index_,i]=s[index_,i]

    '''D'''
    for i in range(n):
        d[i,i]=sum(w[i,:])
    return w,d

def fold(matrix, mode, shape):
    """ Fold a 2D array into a N-dimensional array."""
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    tensor = torch.from_numpy(np.moveaxis(np.reshape(matrix, full_shape), 0, mode))
    return tensor

def unfold(tensor,mode):
    """ Unfolds N-dimensional array into a 2D array."""
    t2=tensor.transpose(mode,0)
    matrix = t2.reshape(t2.shape[0], -1)
    return matrix

def kmode_product(tensor,matrix,mode):
    """ Mode-n product of a N-dimensional array with a matrix."""
    ori_shape=list(tensor.shape)
    new_shape=ori_shape
    new_shape[mode-1]=matrix.shape[0]
    result=fold(np.dot(matrix,unfold(tensor,mode-1)),mode-1,tuple(new_shape))
    return result
def getyi_yj(Y,W):
    l=len(Y)
    re=np.zeros(np.dot(Y[0],Y[0].T).shape)
    for i in range(l):
        for j in range(i+1,l):
            if W[i][j]!=0:
                re=re+np.dot((Y[i]-Y[j]),(Y[i]-Y[j]).T)*W[i][j]*2
    return re

def getyy(Y,D):
    l=len(Y)
    re=np.zeros(np.dot(Y[0],Y[0].T).shape)
    for i in range(l):
        re=re+np.dot(Y[i],Y[i].T)*D[i][i]
    return re
            
def getvalvec(left,right,n_dims):
    eig_val, eig_vec = sp.linalg.eig(left,right)#np.linalg.pinv(right),left))    
    sort_index_ = np.argsort(np.abs(eig_val))
    eig_val = eig_val[sort_index_]
    # print("eig_val:", eig_val[:1])
    j = 0
    while eig_val[j] < 1e-12:
        j+=1
    # print("j: ", j)
    sort_index_ = sort_index_[j:j+n_dims]
    eig_val_picked = eig_val[j:j+n_dims]
    # print(eig_val_picked)
    eig_vec_picked = eig_vec[:, sort_index_] 
    return eig_vec_picked

def getU(newshape,k_near,X_train,P,Band,t):
    w,d=dist(X_train,k_near,t)#计算临近点
    non_zero_w = w[w != 0]
    print("w 中的非零元素为：", non_zero_w)
    U1,U2,U3=np.eye(newshape[0],P),np.eye(newshape[1],P),np.eye(newshape[2],Band)
    U1=U1.astype(np.float64)
    U2=U2.astype(np.float64)
    U3=U3.astype(np.float64)
    t_max=5
    l=len(X_train)
    for t in range(t_max):
        y1,y2,y3=[],[],[]
        for i in range(l):
            y=kmode_product(X_train[i],U2,2)
            y=kmode_product(y,U3,3)
            y1.append(unfold(y,0))
        left=getyi_yj(y1,w)  #(9,9)
        right=getyy(y1,d)
        newu1=getvalvec(left,right,newshape[0])
        # print(newu1.dtype)
        
        lie=newu1.shape[1]
        for i in range(lie):
            newu1[:,i]=newu1[:,i]/np.linalg.norm(newu1[:,i])
        U1=newu1.real.T
        
        for i in range(l):
            y=kmode_product(X_train[i],U1,1)
            y=kmode_product(y,U3,3)
            y2.append(unfold(y,1))
        left=getyi_yj(y2,w)  #(9,9)
        right=getyy(y2,d)
        newu2=getvalvec(left,right,newshape[1])
        lie=newu2.shape[1]
        for i in range(lie):
            newu2[:,i]=newu2[:,i]/np.linalg.norm(newu2[:,i])
        U2=newu2.real.T
        
        for i in range(l):
            y=kmode_product(X_train[i],U1,1)
            y=kmode_product(y,U2,2)
            y3.append(unfold(y,2))
        left=getyi_yj(y3,w)  
        right=getyy(y3,d)
        newu3=getvalvec(left,right,newshape[2])
        lie=newu3.shape[1]
        for i in range(lie):
            newu3[:,i]=newu3[:,i]/np.linalg.norm(newu3[:,i])
        U3=newu3.real.T
    return U1,U2,U3


def getU_MTLPP(newshape, k_near, X_train, ci,P, Band, t):
    '''MTLPP'''
    w, d = dist(ci, k_near, t)  # 计算临近点
    non_zero_w = w[w != 0]
    print("w 中的非零元素为：", non_zero_w)
    U1, U2, U3 = np.eye(newshape[0], P), np.eye(newshape[1], P), np.eye(newshape[2], Band)
    U1 = U1.astype(np.float64)
    U2 = U2.astype(np.float64)
    U3 = U3.astype(np.float64)
    t_max = 5
    l = len(X_train)
    for t in range(t_max):
        y1, y2, y3 = [], [], []
        for i in range(l):
            y = kmode_product(X_train[i], U2, 2)
            y = kmode_product(y, U3, 3)
            y1.append(unfold(y, 0))
        left = getyi_yj(y1, w)  # (9,9)
        right = getyy(y1, d)
        newu1 = getvalvec(left, right, newshape[0])
        # print(newu1.dtype)

        lie = newu1.shape[1]
        for i in range(lie):
            newu1[:, i] = newu1[:, i] / np.linalg.norm(newu1[:, i])
        U1 = newu1.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U3, 3)
            y2.append(unfold(y, 1))
        left = getyi_yj(y2, w)  # (9,9)
        right = getyy(y2, d)
        newu2 = getvalvec(left, right, newshape[1])
        lie = newu2.shape[1]
        for i in range(lie):
            newu2[:, i] = newu2[:, i] / np.linalg.norm(newu2[:, i])
        U2 = newu2.real.T

        for i in range(l):
            y = kmode_product(X_train[i], U1, 1)
            y = kmode_product(y, U2, 2)
            y3.append(unfold(y, 2))
        left = getyi_yj(y3, w)
        right = getyy(y3, d)
        newu3 = getvalvec(left, right, newshape[2])
        lie = newu3.shape[1]
        for i in range(lie):
            newu3[:, i] = newu3[:, i] / np.linalg.norm(newu3[:, i])
        U3 = newu3.real.T
    return U1, U2, U3
def log_euclidean(matrix):
    # 计算特征分解
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # 对对角矩阵 Sigma 中的每个元素取对数
    Sigma = np.log(eigenvalues)
    log_Sigma = np.diag(Sigma)
    # 计算 log(C)
    log_C = eigenvectors @ log_Sigma @ eigenvectors.T
    return log_C

def train_test_tensor(image,label,random_idx):
    num_Class = int(max(label.reshape(label.shape[0] * label.shape[1], 1)))
    Height, Width, Band = image.shape
    image = image.astype(float)
    for band in range(Band):
        image[:, :, band] = (image[:, :, band] - np.min(image[:, :, band])) / (
                    np.max(image[:, :, band]) - np.min(image[:, :, band]))
    data = image
    PATCH_SIZE = 9

    '''Divide the training set and test set and randomly select 10% as the training set'''
    [Height, Width, Band] = data.shape
    image_pad = np.zeros((Height + PATCH_SIZE - 1, Width + PATCH_SIZE - 1, Band))
    for band in range(Band):
        mean_value = np.mean(data[:, :, band])
        image_pad[:, :, band] = np.pad(data[:, :, band], int((PATCH_SIZE - 1) / 2), mode='constant',
                                       constant_values=mean_value)
    data_patch_list = []
    label_patch_list = []
    for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] + int((PATCH_SIZE - 1) / 2)):
        for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] + int((PATCH_SIZE - 1) / 2)):
            if label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)] != 0:
                cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2),
                                  PATCH_SIZE)  # 没问题
                data_patch_list.append(torch.from_numpy(cut_patch.transpose(1, 2, 0)))
                label_patch_list.append(label[i - int((PATCH_SIZE - 1) / 2)][j - int((PATCH_SIZE - 1) / 2)])
    # random_idx = np.random.choice(len(data_patch_list), int(0.1 * len(data_patch_list)), replace=False)
    X_train, train_label, X_test, test_label = [], [], [], []
    for m in random_idx:
        X_train.append(data_patch_list[m])
        train_label.append(label_patch_list[m])
    idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
    for m in idx_test:
        X_test.append(data_patch_list[m])
        test_label.append(label_patch_list[m])

    # x_test = []
    # for i in range(len(X_test)):
    #     q = X_test[i]
    #     x_test.append(q.reshape(q.shape[0] * q.shape[1] * q.shape[2]))
    # x_test = torch.tensor([item.detach().numpy() for item in x_test])
    # x_train = []
    # for i in range(len(X_train)):
    #     q = X_train[i]
    #     x_train.append(q.reshape(q.shape[0] * q.shape[1] * q.shape[2]))
    # x_train = torch.tensor([item.detach().numpy() for item in x_train])
    return X_train,train_label,X_test,test_label


def nn_3dim(x_train, train_label,x_test, test_label):
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
def nn_2dim(x_train, train_label,x_test, test_label):
    x_train = np.array(x_train)
    train_label = np.array(train_label)
    x_test= np.array(x_test)
    test_label= np.array(test_label)
    computedClass = []
    D = np.zeros((x_test.shape[0],x_train.shape[0]))
    for i in range(x_test.shape[0]):
        current_block = x_test[i][:]
        for j in range(x_train.shape[0]):
            neighbor_block = x_train[j][:]
            w = current_block-neighbor_block
            d = np.linalg.norm(w)
            D[i,j] = d
    id = np.argsort(D, axis=1)
    count = 0

    computedClass.append(train_label[id[:, 0]][:])
    for w in range(x_test.shape[0]):
        if computedClass[0][w]==test_label[w]:
            count = count+1
    recogRate = count / x_test.shape[0]

    return recogRate


def train_test_2dim(data, gnd):
    data = np.array(data)
    gnd = np.array(gnd)
    sample, band = data.shape
    data = data.astype(float)
    # for sample in range(sample):
    #     data[sample, :] = (data[sample, :] - np.min(data[sample, :])) / (
    #             np.max(data[sample, :]) - np.min(data[sample, :]))

    label = np.unique(gnd)
    x_train = []
    train_label = []
    x_test = []
    test_label = []
    for i in range(len(label)):
        ind = np.where(gnd == label[i])[0]
        nl = len(ind)
        index = np.random.permutation(nl)
        ktrain = round(nl * 0.1)

        Trin = ind[index[:ktrain]]
        Tein = ind[index[ktrain:]]

        tmpTr = data[Trin, :]
        x_train.extend(tmpTr)
        tmpGTr = gnd[Trin]
        train_label.extend(tmpGTr)
        # ========测试样本===========
        tmpTe = data[Tein, :]
        x_test.extend(tmpTe)
        tmpGTe = gnd[Tein]
        test_label.extend(tmpGTe)

    return x_train, train_label, x_test, test_label
