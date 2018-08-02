import tensorflow as tf
import input_data


mnist = input_data.read_data_sets('datasets',one_hot=True)


tf.reset_default_graph()
sess = tf.InteractiveSession()

# 创建占位符
X = tf.placeholder("float", shape=[None, 28, 28, 1])    #原来教程这里是[None, 784]，但是考虑到移植性，并没有采取
Y = tf.placeholder("float", shape=[None, 10])  

# 设置变量，初始化方式与cousera不同
W1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev = 0.1))  
# 前两个参数是过滤器的大小，第三个参数为输入维数，第四个参数为输出维数
b1 = tf.Variable(tf.constant(0.1, shape=[32]))

# 与cousera不同，这里有了偏置；先激活再池化
Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME' ) + b1
Z1 = tf.nn.relu(Z1)

# 池化 2*2
P1 = tf.nn.max_pool(Z1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# 第二层的参数
W2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))

# 卷积、激活
Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME') + b2
Z2 = tf.nn.relu(Z2)

# 池化 2*2
P2 = tf.nn.max_pool(Z2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# 展开
P2 = tf.reshape(P2, [-1, 7*7*64])   
# -1可以自动计算大小，保证总数不变
# 这里还可以用tf.contrib.layers.flatten(P2)

# 全连接层
W_fc1 = tf.Variable((tf.truncated_normal([7*7*64, 1024], stddev = 0.1)))
b_fc1 = tf.Variable(tf.constant(.1, shape=[1024]))
Z3 = tf.nn.relu(tf.matmul(P2, W_fc1)+ b_fc1)    #matmul 相乘



# 第二个全连接层
W_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev = 0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

Z4 = tf.matmul(Z3, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits =Z4))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_ , 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
batch_size = 50
for i in range(1000):
    batch = mnist.train.next_batch(batch_size)
    train_feature = batch[0].reshape([batch_size, 28, 28, 1])
    train_lable = batch[1]
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session = sess, feed_dict = {x:train_feature, y_:train_lable, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session = sess, feed_dict = {x:train_feature, y_:train_lable, keep_prob: 1})

print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images.reshape([-1,28,28,1]), y_:mnist.test.labels, keep_prob:1.0}))
#stddev、constant、trunancted


