import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 1e-2  # 学习率
wd = 1e-2  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias预测样本X是否为数字0
    @param X: n*d 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: d
    @param bias: 1
    @return: n wx+b
    """
    # TODO: YOUR CODE HERE
    n, d = X.shape
    fx = bias + X @ weight
    return fx
    

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
    print(1)
    n, d = X.shape
    print(1)
    haty = np.zeros(n)
    fx = predict(X, weight, bias)
    sig = sigmoid(fx)
    sigxy = sigmoid(fx * Y)
    loss = (1 / n) * np.sum( - np.log(sigxy)) + wd * np.sum(np.square(weight))
    index = 0
    for sigi in sig:
        if sigi > 0.5:
            haty[index] = 1
        else:
            haty[index] = - 1
        index += 1
    w_tidu = - (1 / n) * np.sum((1 - sigxy)) * (Y @ X) + 2 * wd * weight
    b_tidu = - (1 / n) * np.sum((1 - sigxy) * Y )
    weight = weight - w_tidu
    bias = bias - b_tidu
    return haty,loss,weight,bias
        
        
        
    