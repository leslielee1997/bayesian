from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import numpy as np
import os.path
from numpy import *
import pandas as pd
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator

trainpath0 = 'F:/code/bayesian/train/0'
trainpath1 = 'F:/code/bayesian/train/1'
testpath0 = 'F:/code/bayesian/test/0'
testpath1 = 'F:/code/bayesian/test/1'


def getchar(str):
    c_buf = []
    for word in str:
        c_buf.append(int(word))
    return c_buf


def dataread(path):  # 读取文件夹内所有txt数据,生成二维列表返回
    filenames = os.listdir(path)
    dic = []
    count = 0
    for filename in filenames:
        filepath = path + '/' + filename
        buff = np.loadtxt(filepath, dtype='str')
        di = []
        for i in range(32):
            di.append(getchar(buff[i]))
        dic.append(np.array(di).reshape(1, 1024))
        count += 1
        di.clear()
    return dic


def transdata(dic,k,c): #k为特征数,c为样本类别
    l = len(dic)  # 样本数
    k = len(dic[0][0])  # 特征
    data = {}
    for k in range(k):
        buff = []
        for i in range(l):
            buff.append(dic[i][0][k])
        data.update({'x' + str(k + 1): buff})
    for k in range(k):
        buff = []
        for i in range(l):
            buff.append(c)
        data.update({'c': buff})
    return data

def datamerge(dic1,dic2): #dic1与dic2特征数必须一致
    data = {}
    data.update(dic1)
    k = len(dic2)
    for k in range(k-1):
        data['x'+str(k+1)] += dic2['x'+str(k+1)]
    data['c'] += dic2['c']
    return data

def relation():
    rel = []
    for i in range(1024):
        tup = ('x' + str(i + 1), 'c')
        rel.append(tup)
    return rel


data0 = (transdata(dataread(trainpath0),0))

data1 = (transdata(dataread(trainpath1),1))
data = pd.DataFrame(datamerge(data0,data1))
model = BayesianModel(relation())
pe = ParameterEstimator(model, data)
mle = MaximumLikelihoodEstimator(model,data)
print(mle.estimate_cpd('c'))

