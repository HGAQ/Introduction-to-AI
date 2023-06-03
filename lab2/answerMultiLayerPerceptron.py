import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 1.2e-3   # 学习率
wd1 = 6e-5  # L1正则化
wd2 = 3e-5  # L2正则化
batchsize = 64

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    nodes = [StdScaler(mnist.mean_X, mnist.std_X),
            
            Linear(mnist.num_feat, 490), 
            
            relu(),
            
            Linear(490 ,392),
            
            Dropout(),
            
            tanh(),
             
            Linear(392, 294),
            
            tanh(),
            
            Linear(294, 196),
            
            tanh(),
             
            Linear(196, 147),
            
            tanh(),
            
            Linear(147, 98),
            
            BatchNorm(98),
            
            Linear(98, 49),
            
            sigmoid(),
             
            Linear(49, mnist.num_class), 
            LogSoftmax(), 
            NLLLoss(Y)]
    graph=Graph(nodes)
    return graph