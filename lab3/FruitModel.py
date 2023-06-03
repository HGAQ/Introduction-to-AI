import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集
        #self.dataset = minitraindataset() # 用来调试的小训练集

        # 以下内容可根据需要自行修改
        self.token_num = [{}, {}] # token在正负样本中出现次数
        self.token_total_num = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.all_tokens = set()
        self.text_total_num = 0
        self.count()

    def count(self):
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        for text , label in self.dataset:
            label=int(label)
            self.pos_neg_num[label] += 1
            for token in text:
                if token in self.token_num[label]:
                    self.token_num[label][token] = self.token_num[label][token] + 1
                else:
                    self.token_num[label][token] = 1
        self.token_total_num = len(self.all_tokens)
        self.text_total_num = self.dataset.len

            

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        p_pos = self.pos_neg_num[0] / self.text_total_num
        p_neg = self.pos_neg_num[1] / self.text_total_num
        p_bayes_pos = p_pos
        p_bayes_neg = p_neg

        for token in text:
            if token in self.token_num[0]:
                p_token_pos = (self.token_num[0][token] + 1)/(self.pos_neg_num[0] + len(text))
            else:
                p_token_pos = (1)/(self.pos_neg_num[0] + len(text))
            p_bayes_pos *= p_token_pos
            if token in self.token_num[1]:
                p_token_neg = (self.token_num[1][token] + 1)/(self.pos_neg_num[1] + len(text))
            else:
                p_token_neg = (1)/(self.pos_neg_num[1] + len(text))
            p_bayes_neg *= p_token_neg
        if p_bayes_pos > p_bayes_neg:
            return 0
        else:
            return 1
        
            
            
            
            
def buildGraph(dim, num_classes): #dim: 输入一维向量长度， num_classes:分类数
    # TODO: YOUR CODE HERE
    # 填写网络结构，请参考lab2相关部分
    # 填写代码前，请确定已经将lab2写好的BaseNode.py与BaseGraph.py放在autograd文件夹下
    print(dim,num_classes)
    label=np.zeros(num_classes)
    nodes = [
            Linear(dim, round(dim*7/8)),
            relu(),
            Linear(round(dim*7/8), round(dim*6/8)),
            Dropout(),
            Linear(round(dim*6/8), round(dim*5/8)),
            tanh(),
            Linear(round(dim*5/8), round(dim*4/8)),
            BatchNorm(round(dim*4/8)),
            Linear(round(dim*4/8), round(dim*2/8)),
            sigmoid(),
            Linear(round(dim*2/8),num_classes),
            LogSoftmax(),
            NLLLoss(label)
    ]
    graph = Graph(nodes)
    
    return graph

save_path = "model/mlp.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text):
        # TODO: YOUR CODE HERE
        answer=np.zeros(100)
        count=1
        for word in text:
            if word in self.emb:
                answer=np.add(answer,self.emb[word])
                count+=1
        answer = answer/count
        answer = answer.reshape(-1,100)
        return answer
        
        
        # 利用self.emb将句子映射为一个一维向量，注意，同时需要修改训练代码中的网络维度部分
        # 数据格式可以参考lab2


class MLPModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text):
        X = self.embedding(text)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        p = document.count(word)
        q = len(document)
        return math.log10(p / q + 1)
          

    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        p = len(self.document_list)
        q = 1
        for doc in self.document_list:
            tokens = doc["document"]
            if word in tokens:
                q += 1

        return math.log10(p / q)
         
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        return self.tf(word, document) * self.idf(word)

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据TF-IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        
        max_doc_tfidf = 0
        curr_doc_index = 0
        max_doc_index = 0
        curr_doc_tfidf = 0
        for doc in self.document_list:
            doc_token = doc["document"]
            curr_doc_tfidf = 0
            for word in query:
                curr_doc_tfidf += self.tfidf(word, doc_token)
            if curr_doc_tfidf > max_doc_tfidf:
                max_doc_tfidf = curr_doc_tfidf
                max_doc_index = curr_doc_index
            curr_doc_index += 1
        
        max_sent_tfidf = 0
        max_doc=self.document_list[max_doc_index]
        sentence_list=max_doc["sentences"]
        
        for sent_token, sentence in sentence_list:
            curr_sent_tfidf = 0
            for word in query:
                tf = self.tf(word, sent_token)
                q = 1
                for s_t, s in sentence_list:
                    if word in s_t:
                        q += 1
                idf = math.log10(len(sentence_list)/q)
                curr_sent_tfidf += tf * idf
            if curr_sent_tfidf > max_sent_tfidf:
                max_sent = sentence
                max_sent_tfidf = curr_sent_tfidf
        return max_sent
        

        

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "MLP": MLPModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 1e-3   # 学习率
    wd1 = 1e-4  # L1正则化
    wd2 = 1e-4  # L2正则化
    batchsize = 64
    max_epoch = 100
    dp = 0.1

    graph = buildGraph(100, 2) # 维度需要自己修改

    # 训练
    best_train_acc = 0
    dataloader = traindataset() # 完整训练集
    #dataloader = minitraindataset() # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1

            if cnt == batchsize:
                X = np.concatenate(X, 0)
                Y = np.concatenate(Y, 0)
                
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []
                

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)