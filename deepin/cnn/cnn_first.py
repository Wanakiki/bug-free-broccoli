import numpy as np
import h5py
import matplotlib.pyplot as plt

#matplotlib inline
plt.rcParams['figure.figsize'] = (5.0,4.0)   #设置plt的默认大小
plt.rcParams['image.iterpolation'] = 'nearest'
plt.rcParams['image.camp'] = 'gray'

#load_ext autoreload
#autoreload 2

np.random.seed(1)

def zero_pad(X, pad):
    """
    Argument:
    X(m, n_H, n_W, n_C)
    """
    X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
    #constant 表示连续填充 可以指定填充值，默认为零  constant_values = (x,y)
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    W为过滤器的参数，a_slice_prev是原图像的一个切片，进行单步的卷积操作。
    a_slice_prev 输入数据的切片 (f,f,n_C_prev)
    W 过滤器的权重参数矩阵， (f,f,c_C_prev)
    b 偏差参数矩阵 (1,1,1)  

    Z是返回的一个单个数值
    """
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    实现卷积的向前传播
    A_prev，卷积层之前的激活结果，矩阵 (m, n_H_prev, n_W_prev, n_C_prev)
    W，过滤器的权重， (f,f,n_C_prev,n_C)
    b，偏置 (1,1,1,n_C)
    hparameters, 包含"stride"和"pad"的字典

    返回Z, 卷积层的输出结果， (m, n_H, n_W, n_C)
    caches, 换从的数值用于反向传播
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape  #n_C_prev实际上用不到
    (f, f, n_C_prev, n_C) = W.shape     #输出层的层数由W的参数给定
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[1]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    #开始截取片断：
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, c]
                    Z[i,h,w,c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c]) #这里要注意只传入单层
    
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    实现池化层的传播

    A_prev, 输入矩阵  (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters,    包含f和stride的字典
    mode,   池化层的模式，平均或最大

    返回：
    A,  池化层的输出  (m, n_H, n_W, n_C)
    cache,  用于反向传播计算的字典
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev      #层数不变

    A = np.zeros((m,n_H,n_W,n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice_prev)   #可见池化层没有参数
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    cache = (A_prev, hparameters)
    return A, cache

