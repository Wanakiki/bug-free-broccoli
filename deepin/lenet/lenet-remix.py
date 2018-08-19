import numpy as np
import math 
import progressbar
from skimage import io,transform
import os
import glob
import tensorflow as tf

# 设置图片的尺寸
w = 32
h = 32
c = 1

train_path = "mnist/train/"
test_path = "mnist/test/"

def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index, floder in enumerate(label_dir):
        sum = 0
        for img in glob.glob(floder+'/*.png'):
            if(sum == 128):
                break
            print("reading the image:%s"% img)
            image = io.imread(img)
            image = transform.resize(image, (32,32,1))
            images.append(image)
            labels.append(index)
            sum += 1
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype = np.int32)


# 读取训练集和测试集
train_data, train_label = read_image(train_path)
test_data, test_label = read_image(test_path)

# 打乱训练集和测试集
train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label = train_label[train_image_index]

test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]


X = tf.placeholder(tf.float32, shape=[None, 32,32,1], name='X')
Y = tf.placeholder(tf.int32, shape=[None], name='Y')


W1 = tf.get_variable("W1", shape=[5,5,1,6], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b1 = tf.get_variable("b1", shape=[6], initializer=tf.constant_initializer(0.0))
Z1 = tf.add(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding = "VALID"), b1)
A1 = tf.nn.relu(Z1)

P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

W2 = tf.get_variable("W2", shape=[5,5,6,16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b2 = tf.get_variable("b2", shape=[16], initializer=tf.constant_initializer(0.0))
Z2 = tf.add(tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding = "VALID"), b2)
A2 = tf.nn.relu(Z2)

P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

reshaped = tf.reshape(P2, [-1,400])

W_fc1 = tf.get_variable("W_fc1", shape=[400, 120], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc1 = tf.get_variable("b_fc1", shape=[120], initializer=tf.constant_initializer(0.0))
A_fc1 = tf.nn.relu(tf.matmul(reshaped, W_fc1) + b_fc1)

W_fc2 = tf.get_variable("W_fc2", shape=[120,84], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc2 = tf.get_variable("b_fc2", shape=[84], initializer=tf.constant_initializer(0.0))
A_fc2 = tf.nn.relu(tf.matmul(A_fc1, W_fc2) + b_fc2)

W_fc3 = tf.get_variable("W_fc3", shape=[84, 10], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc3 = tf.get_variable("b_fc3", shape=[10], initializer=tf.constant_initializer(0.0))
Z_fc3 = tf.matmul(A_fc2, W_fc3) + b_fc3

# loss
cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z_fc3, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean)
correct_prediction = tf.equal(tf.cast(tf.argmax(Z_fc3, 1), tf.int32), Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 获取批次
def get_batch(data, label, batch_size):
    for start_index in range(0, len(data)-batch_size+1, batch_size):
        slice_index = slice(start_index, start_index+batch_size)
        yield data[slice_index], label[slice_index]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_num = 10
    batch_size = 64
    for i in range(train_num):
        train_loss, train_acc, batch_num = 0, 0, 0
        for train_data_batch, train_label_batch in get_batch(train_data, train_label, batch_size):
            _, err, acc = sess.run([train_op, cross_entropy_mean, accuracy], feed_dict={
                                   X: train_data_batch, Y: train_label_batch})
            train_loss += err
            train_acc += acc
            batch_num += 1
        print("train loss %s" % (train_loss/batch_num))
        print("train acc %s" % (train_acc/batch_num))

        test_loss, test_acc, batch_num = 0, 0, 0
        for test_data_batch, test_label_batch in get_batch(test_data, test_label, batch_size):
            err, acc = sess.run([cross_entropy_mean, accuracy], feed_dict={
                X: test_data_batch, Y: test_label_batch})
            test_loss += err
            test_acc += acc
            batch_num += 1
        print("test loss %s" % (test_loss/batch_num))
        print("test acc %s" % (test_acc/batch_num))
