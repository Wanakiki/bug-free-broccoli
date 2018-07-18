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
