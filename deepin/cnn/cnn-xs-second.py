import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('datasets',one_hot=True)


tf.reset_default_graph()
sess = tf.InteractiveSession()

# 创建占位符
x = tf.placeholder("float", shape=[None, 28, 28, 1])    #原来教程这里是[None, 784]，但是考虑到移植性，并没有采取
y_ = tf.placeholder("float", shape=[None, 10])  

# 设置变量，初始化方式与cousera不同
W1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev = 0.1))  
# 前两个参数是过滤器的大小，第三个参数为输入维数，第四个参数为输出维数
b1 = tf.Variable(tf.constant(0.1, [32]))

# 与cousera不同，这里有了偏置；先激活再池化
Z1 = tf.nn.conv2d(x, W1, strides = [1,1,1,1], padding = 'SAME' ) + b1
Z1 = tf.nn.relu(Z1)

# 池化 2*2
P1 = tf.nn.max_pool(Z1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# 第二层的参数
W2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1))
b2 = tf.Variable(tf.constant(0.1, [64]))

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
b_fc1 = tf.Variable(tf.constant(.1, [1024]))
Z3 = tf.nn.relu(tf.matmul(P2, W_fc1)+ b_fc1)    #matmul 相乘

# Droput
keep_prob = tf.placeholder('float')
Z_drop = tf.nn.dropout(Z3, keep_prob)

# 第二个全连接层
W_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev = 0.1))
b_fc2 = tf.Variable(tf.constant(0.1, [10]))

y_conv = tf.matmul(Z_drop, W_fc2) + b_fc2

# stddev、constant、trunancted

