import tensorflow as tf
import input_data
import cnn_utils
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets('datasets', one_hot=True)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.get_variable("W1", [5,5,1,32], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding = 'SAME')
Z1 = tf.nn.relu(Z1)

P1 = tf.nn.max_pool(Z1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

W2 = tf.get_variable("W2", [5,5,32,64], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
Z2 = tf.nn.relu(Z2)

P2 = tf.nn.max_pool(Z2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn = None)
P2 = tf.reshape(P2, [-1, 7*7*64])
#print("P2 shape :",P2.shape)
W_fc1 = tf.get_variable("W_fc1", [7*7*64, 1024], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc1 = tf.get_variable("b_fc1", [1024], initializer=tf.zeros_initializer())

Z3 = tf.add(tf.matmul(P2, W_fc1), b_fc1)
Z3 = tf.nn.relu(Z3)
#相乘是有顺序的

W_fc2 = tf.get_variable("W_fc2", [1024,10], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b_fc2 = tf.get_variable("b_fc2", [10], initializer= tf.zeros_initializer())

Z4 = tf.add(tf.matmul(Z3, W_fc2), b_fc2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z4, labels = Y))

# 初始化一些变量， 方便测试
seed = 0
costs = []
learning_rate = 0.1
num_epochs = 2000
m = mnist.train.images.shape[0]

minibatch_size = 200
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
#print("m is",m)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):
        minibatch_cost = 0
        num_minibatches = 1

        for i in range(num_minibatches):
            batch = mnist.train.next_batch(minibatch_size)
            minibatch_X = batch[0].reshape([minibatch_size, 28, 28, 1])
            minibatch_Y = batch[1]
            #print("shape:", minibatch_X.shape)
            _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
            minibatch_cost += temp_cost / num_minibatches

        if epoch % 10 ==0:
            print("Cost after epoch %i %f"%(epoch, minibatch_cost))
        costs.append(minibatch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate ="+str(learning_rate))
    plt.show()

    predict_op = tf.argmax(Z4, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print(accuracy)
    #train_accuracy = accuracy.eval({X:mnist.train.images.reshape([-1, 28,28,1]), Y:mnist.train.labels})
    test_accuracy = accuracy.eval({X:mnist.test.images.reshape([-1,28,28,1]), Y:mnist.test.labels})
    #print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
