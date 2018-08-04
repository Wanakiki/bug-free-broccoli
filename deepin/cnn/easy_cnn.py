import numpy as np
import math
import input_data_xs
import progressbar

def relu(Z):
    return np.maximum(0, Z)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant')
    return X_pad


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=0)


def conv_setp(a_slice_prev, W, b):
    """
    单步卷积
    """
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    # 对矩阵求和
    Z = float(Z) 
    return Z


def conv2d(A_prev, W, b, hparameters):
    """
    卷积 需要一个字典
    """
    # 获取形状和其他基本信息
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)
    (f, f, n_c_prev, n_c) = np.shape(W)
    strides = hparameters['strides']
    pad = hparameters['pad']
    n_h = int((n_h_prev - f + 2*pad) / strides + 1)
    n_w = int((n_w_prev - f + 2*pad) / strides + 1)

    A_prev_pad = zero_pad(A_prev, pad)
    assert(A_prev_pad.shape == (8,32,32,1))
    Z = np.zeros((m, n_h, n_w, n_c))  # 初始化结果
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h * strides
                    vert_end = vert_start + f
                    horiz_start = w * strides
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start: vert_end,
                                         horiz_start: horiz_end, :]
                    Z[i, h, w, c] = conv_setp(a_slice, W[:, :, :, c], b[:, :, :, c])
    return Z


def max_pool(A_prev, hparameters):
    """
    池化层,字典需要包括strides 和 f，无填充
    """
    # 获取形状信息
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)
    f = hparameters['f']
    strides = hparameters['strides']
    n_c = n_c_prev
    n_h = int((n_h_prev-f) / strides + 1)
    n_w = int((n_w_prev - f) / strides + 1)

    Z = np.zeros((m, n_h, n_w, n_c))

    # 要注意池化层只有一片
    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h * strides
                    vert_end = vert_start + f
                    horiz_start = w * strides
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start:vert_end,
                                          horiz_start:horiz_end, c]
                    Z[i, h, w, c] = np.max(a_prev_slice)
    return Z


def conv_back(dZ, cache):
    """
    卷积层的反向传播,
    (A_prev, W, b, hparameters) = cache
    """
    (A_prev, W, b, hparameters) = cache
    (f, f, n_c_prev, n_c) = np.shape(W)
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev
    (m, n_h, n_w, n_c) = np.shape(dZ)
    pad = hparameters['pad']
    strides = hparameters['strides']

    #初始化一些变量
    dA_prev = np.zeros((m, n_h_prev, n_w_prev, n_c_prev))
    dW = np.zeros((f, f, n_c_prev, n_c))
    db = np.zeros((1, 1, 1, n_c))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h * strides
                    vert_end = vert_start + f
                    horiz_start = w * strides
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start: vert_end,
                                         horiz_start: horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    # 截去边缘就很秀
    dA_prev[i, :, :, :] = dA_prev_pad[i, pad[0]:-pad[1], pad[2]: -pad[3], :]
    return dA_prev, dW, db


def pool_back(dA, cache, mode="max"):
    """
    池化层的反向传播，
    (A_prev, hparameters) = cache
    """
    (A_prev, hparameters) = cache
    f = hparameters['f']
    strides = hparameters['strides']
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)
    (m, n_h, n_w, n_c) = np.shape(dA)

    dA_prev = np.zeros((m, n_h_prev, n_w_prev, n_c_prev))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h* strides
                    vert_end = vert_start + f
                    horiz_start = w * strides
                    horiz_end = horiz_start + f

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start: vert_end,
                                              horiz_start:horiz_end, c]
                        mask = a_prev_slice == np.max(a_prev_slice)
                        dA_prev[i, vert_start: vert_end,
                                horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
    return dA_prev

class model:
    np.random.seed(1)
    W1 = np.random.randn(5,5,1,32)
    b1 = np.zeros((1,1,1,32))
    W2 = np.random.randn(14*14*32, 10)
    b2 = np.zeros((1,1,1,10))
    hparameters_conv = {'strides':1, 'pad':2}
    hparameters_pool_max = {'f':2, 'strides':2}

    def __init__(self, learning_rate, batch_size, path, epochs):
        self.learning_rate = learning_rate 
        self.batch_size = batch_size
        self.epochs = epochs
        self.data = input_data_xs.read_data_sets(path, one_hot=True)

        self.mini_batch_train = self.random_batch(self.data[0], self.data[1], batch_size)
    
    def train(self):
        p = progressbar.ProgressBar()
        m = len(self.mini_batch_train)
        index = 0
        for i in p(range(self.epochs)):
            if index >= m:
                index = 0
            self.forward(self.mini_batch_train[index])
            index += 1

    def forward(self, batch):
        self.Z1 = conv2d(batch[0], self.W1, self.b1, self.hparameters_conv)
        self.A1 = relu(self.Z1)
        print(self.A1.shape)

        self.P1 = max_pool(self.A1, self.hparameters_pool_max)
        print(self.P1.shape)
        
        self.P1 = np.reshape(self.P1, [-1, 14*14*32])

        self.Z1 = np.dot(self.P1, self.W2) + self.b2

        self.A2 = softmax(self.Z1)
        self.A3 = np.argmax(self.A2, axis= 1)
        self.A3 = self.A3 == mp.argmax(batch[1], axis = 1)
        print(self.A2.shape)

    def back_ward(self, batch):
        self.dA2 = np.zeros(self.A2.shape)

        for i in range(self.batch_size):
            for j in range(10):
                if self.A3[i][0] == [j]:
                    self.dA2[i][j] = self.A2[i][j] - 1 
                else:
                    self.dA2[i][j] = self.A2[i][j]
        
        
    def random_batch(self, X, Y, mini_batch_size=64, seed=0):
        m = X.shape[0]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :, :, :]
        shuffled_Y = Y[permutation, :]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(m/mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size: k *
                                      mini_batch_size + mini_batch_size, :, :, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k *
                                      mini_batch_size + mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches *
                                      mini_batch_size: m, :, :, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches *
                                      mini_batch_size: m, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


cnn = model(0.1, 8, 'datasets', 500)
cnn.train()
