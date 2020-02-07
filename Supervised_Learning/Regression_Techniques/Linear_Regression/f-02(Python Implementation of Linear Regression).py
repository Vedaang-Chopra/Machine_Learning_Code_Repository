import numpy as np
import pandas as pd
def load():
    from sklearn import datasets
    diab=datasets.load_diabetes()
    x=diab.data
    y=diab.target
    data = np.loadtxt('data.csv', delimiter=",")
    print(data.shape)
    x = data[:, 0]
    y = data[:, 1]
    print(x.shape,y.shape)
    from sklearn import model_selection
    x_train, x_test, y_train, y_test =model_selection.train_test_split(x,y)
    print(x_train.shape,type(x_train))
    # x_train=np.array([[1,2,3,4,5],[7,8,9,10,11]])
    # y_train=np.array([6,12])
    train(x_train,y_train)
    # test(x_test,y_test)

def test(x_test,y_test):
    m, c = fit(x_test, y_test)
    print(m,c)
    y_pred = predict(x_test, m, c)
    cost_test=cost(x_test, y_test, m, c)
    coeff_test=coeff_determination(y_pred, y_test)
    print("Cost=",cost_test," and Score=",coeff_test)

def train(x_train,y_train):
    m,c=fit(x_train,y_train)
    print(m,c)
    y_pred=predict(x_train,m,c)
    # print(y_pred)
    cost_train=cost(x_train,y_train,m,c)
    # print(cost_train)
    coeff_train=coeff_determination(y_pred,y_train)
    print("Cost=", cost_train, " and Score=", coeff_train)

def coeff_determination(y_pred,y_true):
    u=y_true-y_pred
    u=u**2
    u=u.sum()
    v=y_true-y_true.mean()
    v=v**2
    v=v.sum()
    coeff=1-(u/v)
    return coeff


def fit(x_fit,y_fit):
    # For a single column/features of x_train
    # num = (x_fit * y_fit).mean() - (x_fit.mean() * y_fit.mean())
    # den = (x_fit ** 2).mean() - (x_fit.mean() * x_fit.mean())
    # m = num / den
    # c = y_fit.mean() - m * (x_fit.mean())
    # For Multiple columns/features of x_train
    m=np.zeros(x_fit.shape[1])
    for i in range(x_fit.shape[1]):
        num = (x_fit[:,i] * y_fit).mean() - (x_fit[:,i].mean() * y_fit.mean())
        den = (x_fit[:,i] ** 2).mean() - (x_fit[:,i].mean() **2)
        m[i] = num / den
    # for i in range(x_fit.shape[1]):

def predict(x_pred,m,c):
    y_pred=np.zeros(x_pred.shape[0])
    for i in range(x_pred.shape[0]):
        for j in range(x_pred.shape[1]):
            # print(m[j],x_pred[i][j])
            temp=m[j]*x_pred[i][j]
        y_pred[i]=temp+c
    return y_pred


def cost(x,y,m,c):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            temp = m[j] * x[i][j]
        temp = temp + c
        y[i]=y[i]-temp
        y[i]=y[i]**2
    return y[i].sum()

load()

