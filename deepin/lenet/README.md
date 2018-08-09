# LeNet

2018年8月9日

听取了老师的建议，先实现一下LeNet，对MNIST数据集进行识别。毕竟这个就相当于“hello world”一样的存在啊。另外，想要自己实现同样的算法也需要另一个程序来做对比，而且网上关于LeNet的参数设置都可以拿来直接用，省去了不少的麻烦。

这次并没有使用原来的压缩文件进行读取，而是去网上找到了七万张文件的压缩包，本来打算一并上传到Github上，但是七万次的提交直接把电脑搞卡了😓

最终还是放了个压缩包（20Mb)

## 文件说明

1. [lenet-tensorflow.py](lenet-tensorflow.py)：LeNet的tensorflow版本，网上也可以很容易地找到，而且这部分程序也被封装在了tensorflow中。`https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py`
2. [lenet-xs.py]

## 问题

### 激活函数应该在卷积层之后使用，还是应该在池化层之后使用？

这个也算是困扰了几天的问题了，单单从性能上考虑，经过池化之后需要计算的总个数会降低，可能会加快程序的运算。