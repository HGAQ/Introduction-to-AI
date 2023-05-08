import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6

# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
# TODO: You can change or add the hyperparameters here
hyperparams = {"depth":1145, "purity_bound":0.01, "gainfunc":"gainratio"}

def entropy(Y: np.ndarray):
    """
    计算熵
    @param Y: n, 标签向量
    @return: 熵
    """
    n = Y.shape[0]
    uniqued, ncnt = np.unique(Y, return_counts=True)
    S=0
    for cnt in ncnt:
        S += (cnt / n) * (np.log(cnt / n) / np.log(2)) 
    return -S




def gain(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 信息增益
    """
    group = X[:, idx]
    group_size = group.shape[0]
    uniquedgroup, groupcnt = np.unique(group, return_counts=True)
    H_group = entropy(Y)
    H_element=0
    for index in range(uniquedgroup.shape[0]):
        Y_element=[]
        for i in range(group_size):
            if group[i] == uniquedgroup[index]:
                Y_element.append(Y[i])
        H_element += entropy(np.array(Y_element)) * groupcnt[index] / group_size
    H = H_group-H_element
    return H


def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益比
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 信息增益比
    """
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret


def giniD(Y: np.ndarray):
    """
    计算基尼指数
    @param Y: n, 样本的label
    @return: 基尼指数
    """
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]
    return 1 - np.sum(np.multiply(p, p))


def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算负的基尼指数增益
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 负的基尼指数增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        ret -= featp[i] * giniD(Y[mask])
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret


class Node:
    """
    决策树中使用的节点类
    """
    def __init__(self): 
        self.children = {}          # 子节点
        self.featidx: int = None    # 用于划分的特征
        self.label: int = None      # 叶节点的标签

    def isLeaf(self):
        """
        判断是否为叶节点
        @return:
        """
        return len(self.children) == 0


def building(node: Node, X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, purity_bound: float, gainfunc: Callable, prefixstr="" ):
    u, ucnt = np.unique(Y, return_counts=True)
    node.label = u[np.argmax(ucnt)]
    purity = ucnt[np.argmax(ucnt)] / Y.shape[0]
    
    if purity > 1 - purity_bound or depth > hyperparams["depth"] or Y.shape[0]<=1 or len(unused) == 0:
        print(prefixstr, f"label {node.label} numbers {u} count {ucnt}")
        print(purity,depth,Y.shape[0],len(unused))
        return node
    else:
        gains = [gainfunc(X, Y, i) for i in unused]
        idx = np.argmax(gains)
        node.featidx = unused[idx]
        print(prefixstr, f"label {node.label} numbers {u} count {ucnt} id{node.featidx}")
        unused = deepcopy(unused)
        unused.pop(idx)
        feat = X[:, node.featidx]
        ufeat = np.unique(feat)
        for element in ufeat:
            child=Node()
            select=[]
            for i in range(Y.shape[0]):
                if feat[i] == element:
                    select.append(i)
            X_element=X[select,:]
            Y_element=Y[select]
            node.children[element]=child
            building(child, X_element, Y_element, unused, depth+1, purity_bound, gainfunc, prefixstr="")




def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, purity_bound: float, gainfunc: Callable, prefixstr=""):
    root = Node()
    building(root, X, Y, unused, 0, purity_bound, gainfunc,prefixstr)
    # print(prefixstr, f"label {root.label} numbers {u} count {ucnt}") #可用于debug
    # 当达到终止条件时，返回叶节点
    # TODO: YOUR CODE HERE
    # 按选择的属性划分样本集，递归构建决策树
    # 提示：可以使用prefixstr来打印决策树的结构
    # TODO: YOUR CODE HERE
    
    return root


def inferTree(root: Node, x: np.ndarray):
    """
    利用建好的决策树预测输入样本为哪个数字
    @param root: 当前推理节点
    @param x: d*1 单个输入样本
    @return: int 输入样本的预测值
    """
    if root.isLeaf():
        return root.label
    child = root.children.get(x[root.featidx], None)
    return root.label if child is None else inferTree(child, x)

