import numpy as np
import matplotlib.pyplot as plt
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("datasets", one_hot= True)
# 简单的神经网络对手写数字进行识别

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32, [None,10]) # y lable

cost = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y), reduction_indices=[1]))
cost = tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_)   #有这个函数

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

init = tf.global_variables_initializer()  #官网信息有点老旧，最好这样全局初始化

sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #感觉第二个argmax可以不用

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels}))



#fig, ax = plt.subplots(
#   nrows=1,
#     ncols=1,
#     sharex=True,
#     sharey=True, )
# batch_ys = np.argmax(batch_ys, axis=1)
# batch_ys.reshape(100,1)
# print(batch_ys)
# print(batch_ys.shape)
# # for i in range(25):
# #     img = batch_xs[[batch_ys == 7][i]].reshape(28, 28)
# #     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
# for i in range(100):
#     if batch_ys[i] == 7:
#         img = batch_xs[i].reshape(28,28)
#         ax.imshow(img, cmap='Greys', interpolation='nearest')


# plt.tight_layout()
# plt.show()
