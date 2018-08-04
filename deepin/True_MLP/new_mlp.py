import numpy as np
import time
import progressbar
import matplotlib.pyplot as plt
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def dsigmoid(X):
    return (1-X)*X

def softmax(X):
    return  np.exp(X) / np.sum(np.exp(X), axis = 0)

def get_max_value(martix):
    res_list = []
    for j in range(len(martix[0])):
        one_list = []
        for i in range(len(martix)):
            one_list.append((martix[i][j]))
        res_list.append(one_list.index(max(one_list)))
    return res_list
class MLP:
    error = []
    acc = []
    parameters = {}
    caches = {}
    result = []
    def  __init__(self, learning_rate, batch_size, node_nums, train_path, test_path, epoches, mu ,sigma ):
        self.node_nums = node_nums
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.train_data = self.read_data(train_path, True)
        #修改了node——nums
        self.test_data = self.read_data(test_path)

        self.initialize_parameters(mu, sigma)  #初始化只需要node_nms  此处无传参, 这个函数内部更新了 L(层数，包含输入层)
        #随机抽样, 并且只需要对训练集进行操作
        self.mini_batches_train = self.random_batch(self.train_data["X"], self.train_data["Y"], batch_size)

    def train(self):
        p = progressbar.ProgressBar()
        flag_i = -1
        for i in p(range(self.epoches)):
            for batch in self.mini_batches_train:
                self.L_model_forward(batch)
                if(i != flag_i):
                    self.L_model_backward(batch, True)
                    flag_i = i
                else:
                    self.L_model_backward(batch)
                self.update()
            self.L_model_forward((self.train_data["X"], self.train_data["Y"]))
            self.acc.append(np.sum(self.result == self.train_data["Y"]) / self.train_data["Y"].shape[1])

    def predict(self):
        plt.plot(self.acc)
        plt.title("acc-of-trainset  :rate:"+str(self.learning_rate)+" batch_size:"+str(self.batch_size)+" epoches:"+str(self.epoches)+" node_nums:"+str(self.node_nums))
        plt.show()
        plt.plot(self.error)
        plt.title("error of the first bitch")
        plt.show()
        print("=============最终结果(训练集在先)=================")
        self.L_model_forward((self.train_data["X"], self.train_data["Y"]), True)
        self.L_model_forward((self.test_data["X"], self.test_data["Y"]), True)

    def update(self):
        for l in range(1, self.L):
            self.parameters["W"+str(l)] -= self.learning_rate * self.caches["dW"+str(l)]
            if l != self.L -1 :
                self.parameters["b"+str(l)] -= self.learning_rate*self.caches["db"+str(l)]
        #print("finish update")

    def dz_softmax(self,X,batch):
        Y = batch[1]
        dz = X
        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                if i == Y[0][j]:
                    dz[i][j] -=1
        return dz

    def L_model_backward(self, batch,flag = False):
        m = batch[1].shape[1]
        self.caches["dZ"+str(self.L-1)] = self.dz_softmax(self.caches["A"+str(self.L-1)],batch)
        if flag:
            self.error.append((self.caches["dZ"+str(self.L-1)]**2).sum() / m)  #每个循环次统计一次，不然图像太紧
        for l in reversed(range(1, self.L)):
            self.caches["dW"+str(l)] = np.dot(self.caches["dZ"+str(l)],self.caches["A"+str(l-1)].T) / m # dW同样需要除m
            if l != self.L-1:
                self.caches["db"+str(l)] = np.sum(self.caches["dZ"+str(l)]) / m
            if l != 1:
                self.caches["dA"+str(l-1)] = np.dot(self.parameters["W"+str(l)].T, self.caches["dZ"+str(l)])
                self.caches["dZ"+str(l-1)] = dsigmoid(self.caches["A"+str(l-1)])*self.caches["dA"+str(l-1)]
        #print("反向之后：",len(self.caches))
    

        


    def L_model_forward(self, batch, flag = False):
        self.caches["A0"] = batch[0]
        for l in range(1, self.L):
            self.caches["Z"+str(l)] = np.dot(self.parameters["W"+str(l)],self.caches["A"+str(l-1)]) + self.parameters["b"+str(l)]
            if l != self.L -1:
                self.caches["A"+str(l)] = sigmoid(self.caches["Z"+str(l)])
            else:
                self.caches["A"+str(l)] = softmax(self.caches["Z"+str(l)])
        self.result = get_max_value(self.caches["A"+str(self.L-1)])
        if flag:
            print("正确率：",np.sum(self.result == batch[1]) / batch[1].shape[1])
        #print("resule:",self.result)
        #print(len(self.caches))
        #print(self.caches["A3"])
        #print(self.caches["Z3"])

    def random_batch(self, X, Y, batch_size):
        np.random.seed(0)
        m = X.shape[1]
        mini_batches = []
        #cousera 思路 进行随机排序
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))

        num_complete_minibatches = int(m/batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*batch_size: (k+1)*batch_size]
            mini_batch_Y = shuffled_Y[:, k*batch_size: (k+1)*batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % batch_size != 0:
            mini_batch_X = shuffled_X[ :,num_complete_minibatches * batch_size:]
            mini_batch_Y = shuffled_Y[ :,num_complete_minibatches * batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches




    def initialize_parameters(self, mu, sigma):
        np.random.seed(0)
        self.L = len(self.node_nums)
        #parameters = {}
        #cousera  从1开始循环，长度数组就不用考虑了/dz
        for l in range(1, self.L):
            #print("初始化：",l)
            self.parameters['W'+str(l)] = sigma*np.random.randn(self.node_nums[l], self.node_nums[l-1]) + mu
            self.parameters['b'+str(l)] = np.zeros((self.node_nums[l], 1))
            #目前遇到的都需要初始化，尽管softmax不需要偏置，现在初始化为零之后不更新就可以了
        #print(len(self.parameters))

    def read_data(self, path, first_ = False):
        file = open(path)
        lines = file.readlines()
        rows = len(lines)
        Y = np.zeros((1, rows))

        #统计信息
        if(first_):
            self.label_list = []
            for line in lines:
                if line[0] not in self.label_list:
                    self.label_list.append(line[0])
            self.label_num = len(self.label_list)   #为了统一标准只能修改一次label_list label_num
        row = 0
        for line in lines:
            line = line.split('\t')
            if(row == 0):
                features = len(line) - 1 #统计特征数,这个数字实际上也相同
                X = np.zeros((features, rows))

            for i in range(self.label_num):
                if(line[0] == self.label_list[i]):
                    Y[:, row] = i
                    break
            X[:,row] = line[1:]
            row +=1

        data = {"X":X, "Y":Y}
        if(first_):
            self.node_nums.insert(0, features)
            self.node_nums.append(self.label_num)
        return data

#自定义隐藏层的节点数
# __init__(self, learning_rate, batch_size, node_nums, train_path, test_path, epoches, mu ,sigma ):
my_mlp = MLP(0.1, 10, [40,40],"UCI_test.txt","UCI.txt",4000, 0, 0.01)
my_mlp.train()
my_mlp.predict()
