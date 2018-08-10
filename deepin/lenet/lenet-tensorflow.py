from skimage import io,transform
import os
import glob
import numpy as np
import tensorflow as tf

# 设置图片的尺寸
w = 32
h = 32
c = 1

train_path = "mnist/train/"
test_path = "mnist/test/"

# 读取图片和标签的函数
def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index, floder in enumerate(label_dir):
        for img in glob.glob(floder+'/*.png'):
            print("reading the image:%s"%img)
            image = io.imread(img)
            image = transform.resize(image, (w,h,c))
            images.append(image)
            labels.append(index)
        break
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


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





X = tf.placeholder(tf.float32, shape = [None, 32, 32, 1], name='X')
Y = tf.placeholder(tf.int32, shape = [None], name='Y')

W1 = tf.get_variable("W1", shape=[5,5,1,6], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b1 = tf.get_variable("b1", [6], initializer=tf.zeros_initializer())
Z1 = tf.add(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='VALID'), b1) #开始的地方图片被转化为了32*32*1
A1 = tf.nn.relu(Z1)

P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'SAME')    #不使用平均池化

W2 = tf.get_variable("W2", shape=[5,5,6,16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b2 = tf.get_variable("b2", shape=[16], initializer=tf.zeros_initializer())
Z2 = tf.add(tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='VALID'), b2)
A2 = tf.nn.relu(Z2)

P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 还没有看论文，网上大部分的说法是用120个卷积核达到展开的作用
# 但是我这里采取了看到的感觉比较靠谱的做法，先展开，再从400到120增加一个全连接
pool_shape = P2.get_shape().as_list()
nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
reshaped = tf.reshape(P2, [-1, nodes])

# 开始全连接
W_fc1 = tf.get_variable("W_fc1", [400, 120], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc1 = tf.get_variable("b_fc1", [120], initializer=tf.zeros_initializer())

Z4 = tf.add(tf.matmul(reshaped, W_fc1), b_fc1)
A4 = tf.nn.relu(Z4)

W_fc2 = tf.get_variable("W_fc2", [120, 84], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc2 = tf.get_variable("b_fc2", [84], initializer=tf.zeros_initializer())
Z5 = tf.add(tf.matmul(A4, W_fc2), b_fc2)
A5 = tf.nn.relu(Z5)

W_fc3 = tf.get_variable("W_fc3", [84,10], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc3 = tf.get_variable("b_fc3", [10], initializer=tf.zeros_initializer())
Z6 = tf.add(tf.matmul(A5, W_fc3), b_fc3)
# 最后一下没有relu

Y_reshape = tf.reshape(Y, (64,1))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z6, labels=Y_reshape))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.cast(tf.argmax(Z6, 1), tf.int32), Y_reshape)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 获取批次
def get_batch(data, label, batch_size):
    for start_index in range(0, len(data)-batch_size + 1, batch_size):
        slice_index = slice(start_index, start_index + batch_size)
        yield data[slice_index], label[slice_index]


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    train_num = 10
    batch_size = 64

    for i in range(train_num):
        train_loss, train_acc, batch_num = 0, 0, 0
        for train_data_batch, train_label_batch in get_batch(train_data, train_label, batch_size):
            _, err, acc = sess.run([train_op, cross_entropy, accuracy], feed_dict={
                                   X: train_data_batch, Y: train_label_batch})
            train_loss += err
            train_acc += acc
            batch_num += 1
        print("train loss %s"%(train_loss/batch_num))
        print("train acc %s"%(train_acc/batch_num))

        test_loss, test_acc, batch_num = 0,0,0
        for test_data_batch, test_label_batch in get_batch(test_data, test_label, batch_size):
            err, acc = sess.run([cross_entropy, accuracy], feed_dict={
                                   X: test_data_batch, Y: test_label_batch})
            test_loss += err
            test_acc += acc
            batch_num += 1
        print("test loss %s" % (test_loss/batch_num))
        print("test acc %s" % (test_acc/batch_num))
