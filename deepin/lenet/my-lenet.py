import numpy as np
import math
import progressbar
from skimage import io, transform
import os
import glob


# 设置图片的尺寸

# 读取图片和标签的函数


def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index, floder in enumerate(label_dir):
        for img in glob.glob(floder+'/*.png'):
            print("reading the image:%s" % img)
            image = io.imread(img)
            image = transform.resize(image, (32, 32, 1))
            images.append(image)
            labels.append(index)
        break
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def get_batch(data, label, batch_size):
    for start_index in range(0, len(data)-batch_size + 1, batch_size):
        slice_index = slice(start_index, start_index + batch_size)
        yield data[slice_index], label[slice_index]



def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=1)


def zero_pad(X, pad):
    return  np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')


def relu(Z):
    return np.maximum(0, Z)


def d_relu(Z):
    dZ = np.zeros(Z.shape)
    dZ = Z > 0
    #print(dZ)
    return dZ


def conv_setp(a_slice_prev, W, b):
    """
    单步卷积
    """
    return np.sum(np.multiply(a_slice_prev, W) + b)


def conv2d(A_prev, W, b, stride, pad):
    """
    卷积 需要一个字典
    """
    # 获取形状和其他基本信息
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)
    (f, f, n_c_prev, n_c) = np.shape(W)
    n_h = int((n_h_prev - f + 2*pad) / stride + 1)
    n_w = int((n_w_prev - f + 2*pad) / stride + 1)

    A_prev_pad = zero_pad(A_prev, pad)
    #assert(A_prev_pad.shape == (8,32,32,1))
    Z = np.zeros((m, n_h, n_w, n_c))  # 初始化结果
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start: vert_end,
                                         horiz_start: horiz_end, :]
                    Z[i, h, w, c] = conv_setp(
                        a_slice, W[:, :, :, c], b[:, :, :, c])
    return Z


def max_pool(A_prev, strides, f):
    """
    池化层,字典需要包括strides 和 f，无填充
    """
    # 获取形状信息
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)
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
    (A_prev, W, b, pad, strides) = cache
    """
    (A_prev, W, b, pad, strides) = cache
    (f, f, n_c_prev, n_c) = np.shape(W)
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    (m, n_h, n_w, n_c) = np.shape(dZ)
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
        dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad: -pad, :]
    return dA_prev, dW, db


def pool_back(dA, cache, mode="max"):
    """
    池化层的反向传播，
    (A_prev, f, strides) = cache
    """
    (A_prev, f, strides) = cache
    (m, n_h_prev, n_w_prev, n_c_prev) = np.shape(A_prev)
    (m, n_h, n_w, n_c) = np.shape(dA)

    dA_prev = np.zeros((m, n_h_prev, n_w_prev, n_c_prev))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    vert_start = h * strides
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
    W1 = np.random.randn(5,5,1,6)
    b1 = np.zeros((1,1,1,6))
    pad = 0
    stride = 1
    f = 2
    stride = 2
    W2 = np.random.randn(5,5,6,16)
    b2 = np.zeros((1,1,1,16))

    W_fc1 = np.random.randn(400,120)
    b_fc1 = np.zeros((120,1))
    W_fc2 = np.random.randn(120,84)
    b_fc2 = np.zeros((84,1))
    W_fc3 = np.random.randn(84,10)
    b_fc3 = np.zeros((10,1))

    acc = []

    def __init__(self, learning_rate):
        
        self.learning_rate = learning_rate

        train_path = "mnist/train/"
        test_path = "mnist/test/"

        
        # 读取训练集和测试集
        train_data, train_label = read_image(train_path)
        test_data, test_label = read_image(test_path)

        # 打乱训练集和测试集
        train_image_num = len(train_data)
        train_image_index = np.arange(train_image_num)
        np.random.shuffle(train_image_index)
        self.train_data = train_data[train_image_index]
        self.train_label = train_label[train_image_index]

        test_image_num = len(test_data)
        test_image_index = np.arange(test_image_num)
        np.random.shuffle(test_image_index)
        self.test_data = test_data[test_image_index]
        self.test_label = test_label[test_image_index]


    def train_test(self):
        train_num = 10
        batch_size = 64
        
        for i in range(train_num):
            train_acc, batch_num = 0,0
            for train_data_batch, train_label_batch in get_batch(self.train_data, self.train_label, batch_size):
                acc = self.forward(train_data_batch, train_label_batch)
                train_acc += acc
                batch_num += 1
                self.back_ward(train_data_batch)
                self.update(self.learning_rate)
            print("train acc %s" % (train_acc/batch_num))

        test_acc, batch_num = 0, 0
        for test_data_batch , test_label_batch in get_batch(self.test_data, self.test_label, batch_size):
            acc = self.forward(test_data_batch, test_label_batch)
            test_acc += acc
            batch_num += 1
        print("test acc %s" % (test_acc/batch_num))


    def forward(self, data_batch, label_batch):
        self.Z1 = conv2d(data_batch, self.W1, self.b1, 1, 0)
        self.A1 = relu(self.Z1)

        self.P1 = max_pool(self.A1, 2, 2)
        
        self.Z2 = conv2d(self.P1, self.W2, self.b2, 1, 0)
        self.A2 = relu(self.Z2)

        self.P2 = max_pool(self.A2, 2, 2)

        self.P_fc = np.reshape(self.P2, [-1, 400])


        self.Z3 = np.dot(self.P_fc, self.W_fc1) +  self.b_fc1
        self.A3 = relu(self.Z3)

        self.Z4 = np.dot(self.A3, self.W_fc2) + self.b_fc2
        self.A4 = relu(self.Z4)

        self.Z5 = np.dot(self.A4, self.W_fc3) + self.b_fc3
        self.A5 = softmax(self.Z5)

        self.A6 = np.argmax(self.A5, axis=1)
        acc = np.sum( self.A6 == label_batch) / len(label_batch)
        return acc
    
    def back_ward(self, train_data_batch):
        self.dZ5 = self.A5
        for i in range(64):
            for j in range(10):
                if self.A6[i] == j:
                    self.dZ5[i][j] -= 1
        
        self.dW_fc3 = np.dot(self.A4.T, self.dZ5) / 64
        self.db_fc3 = np.sum(self.dZ5) / 64
        self.dA4 = np.dot(self.dZ5, self.W_fc3.T)

        self.dZ4 = self.dA4 * d_relu(self.A4)
        self.dW_fc2 = np.dot(self.A3.T, self.dZ4) / 64
        self.db_fc2 = np.sum(self.dZ4) /64
        self.dA3 = np.dot(self.dZ4, self.W_fc2.T)

        self.dZ3 = self.dA3 * d_relu(self.A3)
        self.dW_fc1 = np.dot(self.A2.T, self.dZ3) /64
        self.db_fc1 = np.sum(self.dZ3) / 64
        self.dP_fc = np.dot(self.dZ3, self.W_fc1.T)

        # 转化型状
        self.dP = self.dP_fc.reshape(64, 5, 5 , 16)
        self.dA2 = pool_back(self.dP, (self.A2, 2,2))
        self.dZ2 = self.dA2 * d_relu(self.A2)

        self.dP1, self.dW2, self.db2 = conv_back(self.dZ2, (self.P1, self.W2, self.b2, 0, 1))
        self.dA1 = pool_back(self.dP1, (self.A1, 2,2))
        self.dZ1 = self.dA1 * d_relu(self.A1)

        _, self.dW1, self.db1 = conv_back(self.dZ1, (train_data_batch, self.W1, self.b1, 0, 1))

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W_fc1 -= self.W_fc1 * learning_rate
        self.b_fc1 -= self.b_fc1 * learning_rate
        self.W_fc2 -= self.W_fc2 * learning_rate
        self.b_fc2 -= self.b_fc2 * learning_rate
        self.W_fc3 -= self.W_fc3 * learning_rate
        self.b_fc3 -= self.b_fc3 * learning_rate


lenet = model(0.01)
lenet.train_test()