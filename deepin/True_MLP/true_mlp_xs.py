import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def dsigmoid(Y):
    return Y * (1 - Y)

def softmax(Y):
    return np.exp(Y) / np.sum(np.exp(Y), axis = 0)

def initialize_parameters(node_nums):
    parameters = {}
    L = len(node_nums)

    for l in range(1, L):
        # 直接实现正态分布 mu = 0, sigma = 0.01
        parameters['W' + str(l)] = 0.01 * np.random.randn(node_nums[l], node_nums[l-1])
        parameters['b' + str(l)] = 0.01 * np.random.randn(node_nums[l], 1)
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


def get_max_value(martix):
    '''
    得到矩阵中每一列最大的值的坐标
    '''
    res_list=[]
    #res_ = []
    for j in range(len(martix[0])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append((martix[i][j]))
        res_list.append(one_list.index(max(one_list)))
        #res_.append(max(one_list))
    #print("查找到的最大值：",res_)
    return res_list

def L_model_backward(AL, Y, caches, label_list):
    grads = {}
    L = len(caches) #少一个参数
    m = AL.shape[1] #总数
    Y = Y.reshape(AL.shape)

    for
    dAL =
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

    print("AL is:",AL.shape,AL)
    result = get_max_value(AL)
    return Y == result



def MLP(learning_rate, batch_size, node_nums, train_path, test_path, epoches):
    train_data, label_list = Read_data(train_path, node_nums, True)
    test_data = Read_data(test_path, node_nums)
    parameters = initialize_parameters(node_nums)
    L = len(node_nums) -1
    #AL, caches = L_model_forward(train_data['X'], parameters, L);
    result = L_model_forward(train_data['X'], train_data['Y'], parameters, L);
    print(node_nums)
    print(train_data)
    print(label_list)
    print(parameters['W3'].shape)
    #print(result,np.sum(result))
    print("AL :",AL.shape,"caches:",len(caches))

# 全局
AL = np.random.randn(1,10)
caches = []
MLP(0.1, 10, [40,40], "UCI.txt", "UCI_test.txt",10000)
