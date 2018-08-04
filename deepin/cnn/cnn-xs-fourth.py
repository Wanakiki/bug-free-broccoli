import numpy as np
import math
import input_data_xs

def relu(Z):
    return np.maximum(0,Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def zero_pad(X,pad):
    X_pad = np.pad(X, ((0,0),(pad[0],pad[1]),(pad[2],pad[3]),(0,0)), 'constant')
    return X_pad

def conv_setp(a_slice_prev, W, b):
    '''
    单步卷积
    '''
    s = np.multiply(a_slice_prev, W)
    # 对矩阵求和
    z = np.sum(s)
    return z + float(b)


def conv2d(A_prev, W, b, strides, padding):
    '''
    卷积
    '''
    # 获取形状和其他基本信息
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)
    (f, f, n_c_prev, n_c) = np.shape(W)
    s_h = strides[1]
    s_w = strides[2]  

    # 目前不清楚tensorflow的SAME VALID算法在遇到向下取整的情况时是怎么处理的
    # 下面为理想型结果：s_h = s_w, f为奇数
    if padding == 'SAME':
        x = (f-1) /2 
        pad = [x,x,x,x]
        n_h = n_h_prev
        n_w = n_w_prev
    elif padding == 'VALID':
        pad = [0,0,0,0]
        n_h = int((n_h_prev - f) / s_h + 1)
        h_w = int((n_w_prev - f) / s_w + 1)

    A_prev_pad = zero_pad(A_prev, pad)
    Z = np.zeros(m, n_h, h_w, n_c)  #初始化结果
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h*s_h
                    vert_end = vert_start + f
                    horiz_start = w * s_w
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start : vert_end, horiz_start: horiz_end, :]
                    Z[i, h, w, c] = conv_setp(a_slice, W[:,:,:,c], b[:,:,:,c])
    return Z

def max_pool(A_prev, ksize, strides, padding = False):
    """
    池化层
    """
    # 获取形状信息
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)

    f_h = ksize[1]
    f_w = ksize[2]
    s_h = strides[1]
    s_w = strides[2]
    n_c = n_c_prev
    
    # 留一块地方以后用来补上padding
    n_h = n_w = int((n_h_prev-f_h) / s_h + 1)

    Z = np.zeros(m, n_h, n_w, n_c)
    

    # 要注意池化层只有一片
    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h*s_h
                    vert_end = vert_start + f_h
                    horiz_start = w * s_w
                    horiz_end = horiz_start + f_w

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    Z[i,h,w,c] = np.max(a_prev_slice)
    return Z

def conv_back(dZ, cache):
    '''
    卷积层的反向传播
    '''
    (A_prev, W, b, pad, strides) = cache
    (f_h, f_w, c_c_prev, n_c) = np.shape(W)
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev
    (m, n_h, n_w, n_c) = np.shape(dZ)
    s_h = strides[1]
    s_w = strides[2]
    
    #初始化一些变量
    dA_prev = np.zeros((m, n_h_prev, n_w_prev, n_c_prev))
    dW = np.zeros((f_h, f_w, n_c_prev, n_c))
    db = np.zeros((1,1,1,n_c))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h*s_h
                    vert_end = vert_start + f_h
                    horiz_start = w * s_w
                    horiz_end = horiz_start + f_w

                    a_slice = a_prev_pad[vert_start : vert_end, horiz_start: horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    # 截去边缘就很秀
    dA_prev[i, :, :, :] = dA_prev_pad[i, pad[0]:-pad[1], pad[2]: -pad[3], :]
    return dA_prev, dW, db

def pool_back(dA, cache, mode = "max"):
    (A_prev, ksize, strides) = cache
    f_h = ksize[1]
    f_w = ksize[2]
    s_h = strides[1]
    s_w = strides[2]
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)
    (m, n_h, n_w, n_c) = np.shape(dA)

    dA_prev = np.zeros(m, n_h_prev, n_w_prev, n_c_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h*s_h
                    vert_end = vert_start + f_h
                    horiz_start = w * s_w
                    horiz_end = horiz_start + f_w

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start: vert_end, horiz_start:horiz_end, c]
                        mask = a_prev_slice == np.max(a_prev_slice)
                        dA_prev[i,vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i,h,w,c]
    return dA_prev



class model:
    np.random.seed(1)
    W1 = np.random.rand(5,5,1,32)
    b1 = np.zeros(32)

    W2 = np.random.rand()
    def __init__(self, learning_rate, batch_size, path, epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.data = input_data_xs.read_data_sets(path, one_hot=True)
    
W1 = np.random.rand(5,5,1,32)
b1 = np.zeros(32)
Z1 = conv2d()
