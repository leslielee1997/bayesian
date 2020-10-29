from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import numpy as np
import os.path
from numpy import *

trainpath0 = 'F:/code/bayesian/train/0'
trainpath1 = 'F:/code/bayesian/train/1'
testpath0 = 'F:/code/bayesian/test/0'
testpath1 = 'F:/code/bayesian/test/1'


def getchar(str):
    c_buf = []
    for word in str:
        c_buf.append(int(word))
    return c_buf


def dataread(path):  # 读取文件夹内所有txt数据
    filenames = os.listdir(path)
    dic = []
    count = 0
    for filename in filenames:
        filepath = path + '/' + filename
        buff = np.loadtxt(filepath, dtype='str')
        di = []  # 单个txt，列表
        for i in range(32):
            di.append(getchar(buff[i]))
        dic.append(np.array(di).reshape(1, 1024))
        count += 1
        di.clear()  # 清空列表
    return dic


def p1(dic): #计算xi=1的概率
    p1 = []
    l = len(dic)
    sum = 0
    for n in range(1024):
        for i in range(l):
            sum += dic[i][0][n]
        p1.append((sum/1024))
        sum = 0
    return  p1

def p0(list): #根据xi=1的概率计算xi=0的概率
    p0 = []
    l =len(list)
    for n in range(l):
        p0.append(1-list[n])
    return p0


def pt(p00,p01,p10,p11):#计算概率表
    pt = [ [0 for i in range(2)] ]
    for i in range(1024):
        pt[0][i] = p00[i]
        pt[0][i] = p01[i]
        pt[1][i] = p10[i]
        pt[1][i] = p11[i]
    return pt

def relation():
    rel = []
    for i in range(1024):
        tup = ('x'+str(i+1),'c')
        rel.append(tup)
    return  rel

def node():
    node = []
    for i in range(1024):
        node.append('x'+str(i+1))
    return  node

def node_val():
    val = []
    for i in range(1024):
        val.append(2)
    return  val

train0 = dataread(trainpath0)
train1 = dataread(trainpath1)
print(len(train0))
'''
p01 = p1(train0)
p00 = p0(p1(train0))
p11 = p1(train1)
p10 = p0(p1(train1))
train_pt = pt(p00,p01,p10,p11)

model = BayesianModel(relation())
print(node())
print(node_val())
class_cpd = TabularCPD(
    variable='c', #节点名称
    variable_card=2, #节点取值个数
    values=train_pt, #节点概率表
    evidence=node(), #该节点的依赖节点
    evidence_card=node_val() #依赖结点的取值个数
)
'''