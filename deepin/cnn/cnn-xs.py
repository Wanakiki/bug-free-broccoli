import numpy as np
import matplotlib.pyplot as plt
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("datasets", one_hot= True)

# x = tf.placeholder("float", [None, 784])
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))

# y = tf.nn.softmax(tf.matmul(x,W)+b)

# y_ = tf.placeholder(tf.float32, [None,10]) # y lable

# #cost = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y), reduction_indices=[1]))
# cost = tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_)   #有这个函数

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# init = tf.global_variables_initializer()  #官网信息有点老旧，最好这样全局初始化

# sess = tf.Session()
# sess.run(init)

# for i in range(1000):
batch_xs, batch_ys = mnist.train.next_batch(100)
# sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})



# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #感觉第二个argmax可以不用

# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels}))

print("batch_xs",batch_xs.shape)
print("batch_ys",batch_ys.shape)