import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_backward(dA, cache):
    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
def softmax_backward(dA, cache):
    return dA


def softmax(Y):
    return np.exp(Y) / np.sum(np.exp(Y), axis = 0)

def initialize_parameters(node_nums):
    parameters = {}
    L = len(node_nums)
    print("初始化长度：",L)
    for l in range(1, L):
        # 直接实现正态分布 mu = 0, sigma = 0.01
        parameters['W' + str(l)] = 0.01 * np.random.randn(node_nums[l], node_nums[l-1])
        parameters['b' + str(l)] = np.zeros((node_nums[l], 1))
        # 初始化最后一层的b 但是不更新

    return parameters

#特征种类 标签种类两个数据集应该一致, 在这个函数内对整个MLP的结构进行补充
def Read_data(path, node_nums, first_ = False):
    file = open(path)
    lines = file.readlines()
    rows = len(lines)
    Y = np.zeros((1, rows))


    label_list = []
    row = 0
    for line in lines:
        if line[0] not in label_list:
            label_list.append(line[0])
    label_num = len(label_list)

    for line in lines:
        line = line.split('\t')
        if(row == 0):
            features = len(line) - 1
            X = np.zeros((features, rows))

        for i  in range(label_num):
            if(line[0] == label_list[i]):
                Y[:,row] = i
                break

        X[:,row] = line[1:]
        row += 1

    if(first_):
        node_nums.insert(0, features)
        node_nums.append(label_num)
    data = {"X": X,
            "Y": Y}
    if(first_):
        return data, label_list
    return data

# A = F(Z) = F(WX + b)
# caches 为缓存区
# linear_cache (A, W, b)
# activation_cache(Z)
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache =(A, W, b)
    return Z, cache

def linear_activation_forward(A_prew, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prew, W, b)
        A, activation_cache = sigmoid(Z), Z

    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prew, W, b)
        A, activation_cache = softmax(Z), Z

    cache = (linear_cache, activation_cache)
    return A, cache

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        db = 0
    return dA_prev, dW, db

def get_max_value(martix):
    '''
    得到矩阵中每一列最大的值的坐标
    '''
    res_list=[]
    #res_ = []
    #print(martix)
    for j in range(len(martix[0])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append((martix[i][j]))
        res_list.append(one_list.index(max(one_list)))
        #res_.append(max(one_list))
    #print("查找到的最大值：",res_list)
    return res_list

def L_model_backward(AL, result_lable, Y, result, caches, label_list):
    global error
    L = len(caches) #少一个参数
    m = AL.shape[1] #总数
    Y = Y.reshape((1,m))

    dAL = np.zeros(AL.shape)
    #print(result)
    for j in range(m):
        for i in range(len(label_list)):
            if (i == Y[0][j]):
                dAL[i][j] = AL[i][j] - 1
            else:
                dAL[i][j] = AL[i][j]

    error = np.sum(dAL**2) / m
    print("error is:",error)
    dZL = dAL
    #print(dZL)
    current_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, "softmax")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, "sigmoid")
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
    #print(grads["db1"])

    return grads

def update(parameters, grads, learning_rate, L):
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        if l != L-1:
            parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]

    return parameters
def L_model_forward(X, Y, parameters, L):
    global AL
    global caches
    caches = []
    A = X
    for l in range(1, L):
        A_prew = A
        A, cache = linear_activation_forward(A_prew, parameters['W' + str(l)],parameters['b' + str(l)], "sigmoid")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)

    #print("AL is:",AL.shape,AL)
    result = get_max_value(AL)
    print("get _max:",result)
    print("Y:",Y)
    return AL, result,Y == result

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    num_complete_minibatches = int(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size: (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size: (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[ :,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[ :,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def MLP(learning_rate, batch_size, node_nums, train_path, test_path, epoches):
    train_data, label_list = Read_data(train_path, node_nums, True)
    test_data = Read_data(test_path, node_nums)
    mini_batches_train = random_mini_batches(train_data["X"], train_data["Y"], 10)
    L = len(node_nums) -1
    parameters = initialize_parameters(node_nums)
    #print(np.sum(result_lable) / len(test_data["Y"]))
    for i in range(epoches):
        for batch in mini_batches_train:
            AL, result, result_lable = L_model_forward(batch[0], batch[1],parameters,L)
            grads = L_model_backward(AL, result_lable, batch[1], result, caches, label_list)
            parameters = update(parameters, grads, learning_rate, L)
    _1,_, result_lable = L_model_forward(train_data["X"], train_data["Y"],parameters,L)
    print("正确率：",np.sum(result_lable)/300)
    _1,_, result_lable = L_model_forward(test_data["X"], test_data["Y"],parameters,L)
    #print(result_lable)
    print(np.sum(result_lable) / len(test_data["Y"]))


# 全局
AL = np.random.randn(1,10)
grads = {}
caches = []
error = 0
MLP(0.1, 10, [40,40], "UCI.txt", "UCI_test.txt",10)
