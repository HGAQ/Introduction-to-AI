import numpy as np
from numpy.random import randn
import mnist
import pickle
from util import setseed
# 超参数

lr = 0.1  # 学习率
wd = 5e-4  # l2正则化项系数

setseed(0) # 固定随机数种子以提高可复现性
save_path = "model/mymodel.npy"



def predict(X, weight, bias):
    """
    使用输入的weight和bias预测样本X是否为数字0
    @param X: n*d 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: d
    @param bias: 1
    @return: n wx+b
    """
    # TODO: YOUR CODE HERE
    fx = bias + X @ weight
    return fx # 1--n
    

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: n*d 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: d
    @param bias: 1
    @param Y: n 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: n 模型的输出, 为正表示数字为0, 为负表示数字不为0 
        loss: 1 由交叉熵损失函数计算得到
        weight: d 更新后的weight参数
        bias: 1 更新后的bias参数
    """
    'n: 样本数量, d: 样本的维度'
    n, d = X.shape
    haty = predict(X, weight, bias) # 1--n 
    sigxy = sigmoid(haty * Y) # 1--n
    loss = (1 / n) * np.sum( - np.log(sigxy + 1e-6)) + wd * np.sum(np.square(weight))
    w_tidu = - (1 / n) * (((1 - sigxy) * Y ) @ X)  + 2 * wd * weight
    b_tidu = - (1 / n) * np.sum((1 - sigxy)  * Y ) 
    weight = weight - lr * w_tidu
    bias = bias - lr * b_tidu
    return haty,loss,weight,bias



X = mnist.trn_X
Y = mnist.trn_Y 


if __name__ == "__main__":
    # 训练
    weightlist=[]
    biaslist=[]
    for is_num in range(10):
        best_train_acc = 0
        Y_num = 2 * (Y == is_num) - 1
        bestweight = weight = randn(mnist.num_feat)
        bestbias = bias = np.zeros((1))
        for i in range(1, 150+1):
            haty, loss, weight, bias = step(X, weight, bias, Y_num)
            acc = np.average((haty > 0).flatten() == (Y_num > 0))
            print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
            if acc > best_train_acc:
                best_train_acc = acc
                bestbias=bias
                bestweight=weight
        weightlist.append(bestweight)
        biaslist.append(bestbias)
    with open(save_path, "wb") as f:
        pickle.dump((weightlist, biaslist), f)

