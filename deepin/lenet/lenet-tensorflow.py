import tensorflow as tf
import numpy as np


tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape = [None, 10])

W1 = tf.get_variable("W1", shape=[5,5,1,6], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b1 = tf.get_variable("b1", [6], initializer=tf.zeros_initializer())
Z1 = tf.add(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME'), b1) #填充算法为SAME就不需要对输入进行转化
A1 = tf.nn.relu(Z1)

P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'SAME')    #不使用平均池化

W2 = tf.get_variable("W2", shape=[5,5,6,16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b2 = tf.get_variable("b2", shape=[16], initializer=tf.zeros_initializer())
Z2 = tf.add(tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='VALID'), b2)
A2 = tf.nn.relu(Z2)

P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W3 = tf.get_variable("W3", shape=[5,5,16,120], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b3 = tf.get_variable("b3", shape=[120], initializer=tf.zeros_initializer())
Z3 = tf.add(tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding="VALID"), b3)
A3 = tf.nn.relu(Z3)

A3 = tf.reshape(A3, [-1, 120])
print("A3 is ", A3)
W_fc1 = tf.get_variable("W_fc1", [120, 84], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc1 = tf.get_variable("b_fc1", [84], initializer=tf.zeros_initializer())

Z4 = tf.add(tf.matmul(A3, W_fc1), b_fc1)
A4 = tf.nn.relu(Z4)

W_fc2 = tf.get_variable("W_fc2", [84, 10], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc2 = tf.get_variable("b_fc2", [10], initializer=tf.zeros_initializer())
Z5 = tf.add(tf.matmul(A4, W_fc2), b_fc2)
# 最后一下没有relu

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z5))

#训练时会用到的参数↓
seed = 0
costs = []
learning_rate = 0.01
num_epoches = 5000
m = mnist.train.images.shape[0]
minibatch_size = 50
optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epoches):
        minibatch_cost = 0
        num_minibatches = 1 #一个epoch计算的batch数量，就应该是一啊 汗

        for i in range(num_minibatches):
            batch = mnist.train.next_batch(minibatch_size)
            minibatch_X = batch[0].reshape([minibatch_size, 28, 28, 1])
            minibatch_Y = batch[1]

            _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
            minibatch_cost += temp_cost / num_minibatches

        if epoch % 10 ==0:
            print("Cost after epoch %i %f"%(epoch, minibatch_cost))
        costs.append(minibatch_cost)

    predict_op = tf.argmax(Z4, axis = 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, axis = 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print(accuracy)
    train_accuracy = accuracy.eval({X:mnist.train.images.reshape([-1, 28,28,1]), Y:mnist.train.labels})
    test_accuracy = accuracy.eval({X: mnist.test.images.reshape([-1, 28, 28, 1]), Y: mnist.test.labels})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
