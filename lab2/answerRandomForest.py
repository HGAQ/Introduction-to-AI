from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 65     # 树的数量
ratio_data = 0.5   # 采样的数据比例
ratio_feat = 0.5 # 采样的特征比例
hyperparams = {"depth":60, "purity_bound":0.1, "gainfunc":negginiDA} # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    forest = []
    n, d = X.shape
    n_ = round(n * ratio_data)
    d_ = round(d * ratio_feat)
    for i in range(num_tree):
        col_rand_array = np.arange(n)
        line_rand_array = np.arange(d)
        np.random.shuffle(col_rand_array)
        np.random.shuffle(line_rand_array)
        X_rand=X[col_rand_array[0:n_],:]
        Y_rand=Y[col_rand_array[0:n_]]
        unused=list(line_rand_array[0:d_])
        tree=buildTree(X_rand,Y_rand,unused,hyperparams["depth"],hyperparams["purity_bound"],hyperparams["gainfunc"])
        forest.append(tree)
    return forest

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
