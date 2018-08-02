import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import datetime
import input_data

mnist = input_data.read_data_sets("datasets/", one_hot=True)
tf.reset_default_graph()
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 28, 28, 1])  # 后三个代表图片的高度、宽度以及通道数
y_ = tf.placeholder("float", shape=[None, 10])  # 10为类型数
# 5,5 表示过滤器的大小，1表示输入的通道数，32表示过滤器的大小
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(.1, shape=[32]))  # 偏置参数只需要满足通道数相同就
h_conv1 = tf.nn.conv2d(input=x, filter=W_conv1, strides=[
                       1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[
                         1, 2, 2, 1], padding='SAME')


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2_by_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Second Conv and Pool Layers
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(.1, shape=[64]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2_by_2(h_conv2)

# First Fully Connected Layer
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(.1, shape=[1024]))
h_pool2_flat = (h_pool2, [-1, 7 * 7 * 64])  # -1 表示自动计算出合适的值，这里计算结果为1
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Second Fully Connected Layer
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(.1, shape=[10]))

# Final Layer
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
crossEntropyLoss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batchSize = 50
for i in range(1000):
    batch = mnist.train.next_batch(batchSize)
    trainingInputs = batch[0].reshape([batchSize, 28, 28, 1])
    trainingLabels = batch[1]
    if i % 100 == 0:
        trainAccuracy = accuracy.eval(session=sess, feed_dict={
                                      x: trainingInputs, y_: trainingLabels, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, trainAccuracy))
    trainStep.run(session=sess, feed_dict={
                  x: trainingInputs, y_: trainingLabels, keep_prob: 0.5})
