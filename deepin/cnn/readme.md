# CNN相关内容

## 文件说明

Cousera 上的课程没有对CNN的反向传播进行详细的解释，仅仅在课后题中提及，需要花时间理解，纯python实现cnn的[代码](cnn_first.py)，另外还有tensorflow实现的版本。
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