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
