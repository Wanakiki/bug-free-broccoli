import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


index = 6       #改变index可以控制输出不同的图片
plt.imshow(X_train_orig[index])
import pylab
pylab.show()
print ("y = "+str(np.squeeze(Y_train_orig[:,index])))

def creat_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    创建placeholder，为了session

    n_H0, n_W0, n_C0,输入图像的高宽厚度
    n_y 需要辨别的数量
    """
    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape = (None, n_y))

    return X,Y

def initialize_parameters():
    """
    初始化权重，tensorflow会自动添加偏置项，所以只需要初始化W，
    这个程序在cousera上默认的是两层神经网络，W1 : [4, 4, 3, 8]，W2 : [2, 2, 8, 16]

    返回包括W1，W2的字典
    """

    tf.set_random_seed(1)   #自己测试程序的时候最好还是吧随机种子加上
    W1 = tf.get_variable("W1", [4,4,3,8],initializer= tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2,2,8,16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1":W1, "W2":W2}

    return parameters

def forward_propagation(X, parameters):
    """
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    X   输入的数据placeholder，shape:(input size, number of example)
    parameters  字典

    这个函数实际上按照课后题的说明实现了一种可能：
    - Conv2D: stride 1, padding is "SAME"
    - ReLU
    - Max pool: Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
    - Conv2D: stride 1, padding is "SAME"
    - ReLU
    - Max pool: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
    - Flatten the previous output.
    - FULLYCONNECTED (FC) layer: Apply a fully connected layer without an non-linear activation function.
    tensorflow关于softmax还要单独调用

    返回：
    Z3  最后隐藏层的节点输出
    """

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')

    A1 = tf.nn.relu(Z1)

    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')

    A2 = tf.nn.relu(Z2)

    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    P2 = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn = None) #一定要加上None，否则会调用函数
    #因为测试集的图片只有六种数字，所以第二个参数是6

    return Z3

def compute_cost(Z3, Y):
    """
    计算cost，return: cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    #求平均值函数内是一个向量，但是cost是一个数，所以有求平均值的过程
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    懒了懒了/xk
    """

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X,Y = creat_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1 #使每次循环随机不同（minibatch)

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 ==0:
                print ("Cost after epoch %i : %f"%(epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate ="+str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3, 1)   #返回一个轴上最大值的索引
        correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))   #判断两个是否相等 返回0、1

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #计算平均值
        print(accuracy)
        train_accuracy = accuracy.eval({X:X_train, Y:Y_train})  #不懂eval
        test_accuracy = accuracy.eval({X:X_test, Y:Y_test})
        print("Train Accuracy:",train_accuracy)
        print("Test Accuracy:",test_accuracy)
        return train_accuracy,test_accuracy, parameters


X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

_, _, parameters = model(X_train, Y_train, X_test, Y_test)
