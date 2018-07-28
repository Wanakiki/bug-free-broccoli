- 矩阵的某一行 [1]
- 矩阵的某一列 a[:,1]
- 超参数的选区会随着数据集的情况，CPU、GPU的架构的变化而产生变化
- cost:
```python
def compute_cost(A2, Y, parameters):
    log = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    # multiply is very import
    cost = -np.sum(log) / Y.shape[1]
    cost = np.squeeze(cost)

    #assert(isinstance(cost, float))

    return cost
```
- 求和：
```python
db = np.sum(dZ, axis = 1, keepdims = True) / m
```
求和需要keepdims保证前后不变，axis = 1 是行， = 0 为列

- 非标准正态分布:`` Two-by-four array of samples from N(3, 6.25):2.5*np.random.randn(2,4)+3``

- tf.placeholder(dtype, shape=None, name=None)
- W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
- b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
- xavier_initializer(
    uniform=True,
    seed=None,
    dtype=tf.float32
)
- tf.nn.relu(Z1)   

- 插入图片\usepackage{graphicx}
- \includegraphics[scale=0.4]{bias.jpg}


- tensorflow :
  - 初始化： W1 = tf.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

  In TensorFlow, there are built-in functions that carry out the convolution steps for you.

  - **tf.nn.conv2d(X,W1, strides = [1,s,s,1], padding = 'SAME'):** given an input $X$ and a group of filters $W1$, this function convolves $W1$'s filters on X. The third input ([1,f,f,1]) represents the strides for each dimension of the input (m, n_H_prev, n_W_prev, n_C_prev). You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)

  - **tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME'):** given an input A, this function uses a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window. You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)

  - **tf.nn.relu(Z1):** computes the elementwise ReLU of Z1 (which can be any shape). You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/nn/relu)

  - **tf.contrib.layers.flatten(P)**: given an input P, this function flattens each example into a 1D vector it while maintaining the batch-size. It returns a flattened tensor with shape [batch_size, k]. You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten)

  - **tf.contrib.layers.fully_connected(F, num_outputs):** given a the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)

  In the last function above (`tf.contrib.layers.fully_connected`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.
