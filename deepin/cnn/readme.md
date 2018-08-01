# CNN相关内容

## 文件说明

Cousera 上的课程没有对CNN的反向传播进行详细的解释，仅仅在课后题中提及，需要花时间理解，纯python实现cnn的代码：[cnn_first.py](cnn_first.py)，另外还有tensorflow实现的版本：[cnn_tensorflow.py](cnn_tensorflow.py)，tensorflow实现版本依赖本文件夹中的[cnn_utils.py](cnn_utils.py)文件提供一些必须的函数。

另外还在tensorflow中文社区上找到一个有关于MNIST数据集的入门教程，下载了该数据集的压缩文件放到了datasets目录下，因为这个数据集文件读取相对麻烦，直接采用了提供的程序来提取数据，即[input_data.py](input_data.py)，再之后的几个文件中均使用了其中定义的函数。

要注意的一点是input_data.py中的``read_data_sets``函数本身就可以创建文件夹并下载文件读取，只需要自己设置下目录，但不清楚linux与windows系统下方式是否相同。

>MNIST数据集的图片大小为28*28=784，共有0~9十个标签，训练集有60000对数据，测试集有10000对数据。

tensorflow社区代码的抄写（均与手写数字识别相关）：

- 简单的神经网络：[cnn-xs-first.py](cnn-xs-first.py)
- 卷积神经网络：[cnn-xs-second.py](cnn-xs-second.py)


## 一些有用的函数/方程

### 相关概念

#### Pooling layer

The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. The two types of pooling layers are:

- Max-pooling layer: slides an ($f, f$) window over the input and stores the max value of the window in the output.

- Average-pooling layer: slides an ($f, f$) window over the input and stores the average value of the window in the output.
  
### 推导相关

对于卷积层来说：
$$ n_H = \lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$

$$ n_W = \lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$

$$ n_C = \text{number of filters used in the convolution} $$

对于池化层来说，因为没有填充，公式少了一个变量：
$$ n_H = \lfloor \frac{n_{H_{prev}} - f}{stride} \rfloor +1 $$
$$ n_W = \lfloor \frac{n_{W_{prev}} - f}{stride} \rfloor +1 $$
$$ n_C = n_{C_{prev}}$$

### Python函数

- numpy.pad()
    1. 语法结构： pad(array, pad_width, mode, **kwargs)，返回值为数组
    2. 参数解释： array为需要填充的数组；pad_width为每个轴（axis）边缘需要的填充的数值数目； mode表示填充的方式，共有11种。
    3. 填充方式：
        - ‘constant’——表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0edge’——表示用边缘值填
        - ‘linear_ramp’——表示用边缘递减的方式填充
        - ‘maximum’——表示最大值填充
        - ‘mean’——表示均值填充
        - ‘median’——表示中位数填充
        - ‘minimum’——表示最小值填充
        - ‘reflect’——表示对称填充
        - ‘symmetric’——表示对称填充
        - ‘wrap’——表示用原数组后面的值填充前面，前面的值填充后面

- numpy.ndarray.max
    1. 结构：ndarray.max(axis=None, out=None)
    2. 说明：沿给定轴计算矩阵的最大值
- numpy.mean
    1. 结构: ``numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>)``
    2. 说明：沿指定轴的算术平均值；返回数组元素的平均值。
    3. 参数：
        - a  所求数组
        - axis   None或者整数或者由整数组成的元组，表明计算平均值的axis or axes，如果是一个元组，则在多个轴上面计算，而不是a single axis or all the axes
        - dtype 计算平均值的数据类型，可选；整数输入的默认值为float64，浮点数输入与浮点数相同
        - out   放置结果的备用输出数组，可选
        - keepdim   布尔型，可选；若设置为1则将缩小的轴尺寸变为一？