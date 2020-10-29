import numpy as np
import os.path
from numpy import *
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

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
        dic.append(np.array(di))
        count += 1
        di.clear()  # 清空列表
    return dic


def corssFeatureExtract(img):  # 笔划密度降维
    crosspoint = []
    for i in range(0, 32, 4):
        crossPointNumR = 0
        crossPointNumC = 0
        fR = True
        fC = True
        for j in range(31):
            if img[i][j] != img[i][j + 1]:
                fR = bool(1 - fR)
                if fR == True:
                    crossPointNumR += 1
            if img[j][i] != img[j + 1][i]:
                fC = bool(1 - fC)
                if fC == True:
                    crossPointNumC += 1
        crosspoint.append(crossPointNumR)
        crosspoint.append(crossPointNumC)
    return crosspoint


def transdata(dic, k, c):  # k为特征数 16 ,c为样本类别
    l = len(dic)
    data = {}
    buff = []
    clist = []
    for t in range(k):
        for i in range(l):
            buff.append(dic[i][t])  # t：第几维，i第几个样本
    for count in range(k):
        data.update({'x' + str(count + 1): buff[0 + l * count:l + l * count]})
    buff.clear()

    for count1 in range(l):
        clist.append(c)
    data.update({'c': clist[0:]})
    clist.clear()
    return data


def datamerge(dic1, dic2):
    data = {}
    data.update(dic1)
    k = len(dic2)
    for k in range(k - 1):
        data['x' + str(k + 1)] += dic2['x' + str(k + 1)]
    data['c'] += dic2['c']
    return data


def relation():
    rel = []
    for i in range(16):
        tup = ('c', 'x' + str(i + 1))
        rel.append(tup)
    return rel


def predict(di, c):
    l = len(di)
    count = 0
    for i in range(l):
        pr = c_infer.query(
            variables=['c'],
            evidence={'x1': di[i][0], 'x2': di[i][1], 'x3': di[i][2], 'x4': di[i][3], 'x5': di[i][4], 'x6': di[i][5],
                      'x7': di[i][6], 'x8': di[i][7], 'x9': di[i][8], 'x10': di[i][9], 'x11': di[i][10],
                      'x12': di[i][11], 'x13': di[i][12], 'x14': di[i][13], 'x15': di[i][14], 'x16': di[i][15]}
        )
        print('第'+str(i+1)+'个样本预测概率：',(pr))
        if(pr.values[0] > 0.5):
            count += 1
    print('\n给定测试集为:', c)
    print('预测为0率: {:.2f}%'.format(count / l * 100))

if __name__ == "__main__":
    data0 = dataread(trainpath0)
    newdata0 = []
    data1 = dataread(trainpath1)
    newdata1 = []

    for i in range(len(data0)):
        newdata0.append(corssFeatureExtract(data0[i]))
    for i in range(len(data1)):
        newdata1.append(corssFeatureExtract(data1[i]))
    traindata = pd.DataFrame(datamerge(transdata(newdata0, 16, '0'), transdata(newdata1, 16, '1')))

    test0 = dataread(testpath0)
    test1 = dataread(testpath1)
    newtest0 = []
    newtest1 = []
    for i in range(len(test1)):
        newtest1.append(corssFeatureExtract(test1[i]))
    for i in range(len(test0)):
        newtest0.append(corssFeatureExtract(test0[i]))

    model = BayesianModel(relation())
    pe = ParameterEstimator(model, traindata)
    mle = MaximumLikelihoodEstimator(model, traindata)
    # print(pe.state_counts('x1')) #x1的频数
    model.fit(traindata, estimator=MaximumLikelihoodEstimator)
    # print(mle.estimate_cpd('x1')) #x1的cpd
    c_infer = VariableElimination(model)
    predict(newtest0, '0')
