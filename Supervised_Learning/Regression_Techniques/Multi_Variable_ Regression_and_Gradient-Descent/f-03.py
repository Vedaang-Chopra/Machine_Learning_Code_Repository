import numpy as np
import pandas as pd
def cost_calc(x_train,y_train,m):
    # print(m)
    cost,cost0=0,0
    s=0
    c1, c2 = x_train.shape
    # print(c1,c2)
    # print(y_train, x_train)
    cost1 = x_train* m
    cost1=cost1.values
    # print(cost1)
    y_train=y_train.values
    for i in range(0,c1):
        for j in range(0, c2):
            s = s + cost1[i][j]
        cost2 = y_train[i][0] - s
        cost2=cost2**2
        cost0 += cost2
        s=0
    cost = cost0 / len(x_train)
    return cost

def step_grad(x_train,y_train,m,columnno):
    var,var2,s=0,0,0
    c1, c2 = x_train.shape
    var0 = m * x_train
    # print(var0)
    var0=var0.values
    y_train=y_train.values
    x_train=x_train.values
    # print(y_train.shape)
    # print(y_train)
    for i in range(0, c1):
        for j in range(0, c2):
            s = s + var0[i][j]
        var1=y_train[i][0]-s
        var1+=var1*x_train[i][columnno]
        var2+= var1
        s=0
    var = var2 * (-2/len(x_train))
    return var

def grad_descent(x_train,y_train,learning_rate,num_iterations):
    m = np.zeros(x_train.shape[1])
    new_m=m.copy()
    # print(m)
    cost=0
    for i in range(num_iterations):
        for j in range(0,len(m)):
            new_m[j] = step_grad(x_train,y_train,m,j)
        # print(m)
        m = m - new_m * learning_rate
        cost=cost_calc(x_train,y_train,m)
        # print("The Cost for m:",m," is:",cost)
    print("The Cost for m:",m," is:",cost)
    return m,cost


def loading():
    # data = np.loadtxt('data.csv', delimiter=",")
    # # print(data)
    # print(data.shape)
    # x = data[:, 0]
    # x = x.reshape(-1, 1)
    # y = data[:, 1]
    # y = y.reshape(-1, 1)
    from sklearn import datasets
    diab=datasets.load_boston()
    # print(diab)
    x=diab.data
    y=diab.target
    x = np.genfromtxt('G:\Development\Projects\Python Projects\Machine_Learning\Gradient_Descent\Boston_Project\\0000000000002417_training_boston_x_y_train.csv',delimiter=',')
    t = np.genfromtxt('G:\Development\Projects\Python Projects\Machine_Learning\Gradient_Descent\Boston_Project\\0000000000002417_test_boston_x_test.csv',delimiter=',')
    print(x.shape,type(x))
    y = x[:, 13]
    x = x[:, 0:13]
    x=pd.DataFrame(x)
    y=pd.DataFrame(y)
    t=pd.DataFrame(t)
    print(x.shape, type(x))
    print(y.shape, type(y))
    print(t.shape,type(t))
    x[(x.shape[1]+1)]=1
    print(x.shape, type(x))
    # print(x)
    from sklearn import model_selection
    x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3)
    # print(x_train)
    learning_rate=0.000002
    num_iterations=100
    # print(x_train)
    # print(type(x_train))
    # x_train=np.array([[1,2,3,4,5,1],[1,1,1,1,1,1]])
    # y_train=np.array([[6],[7]])
    m1,cost1=grad_descent(x_train,y_train,learning_rate,num_iterations)
    print("Testing Data.............................")
    m2,cost2= grad_descent(x_test, y_test, learning_rate, num_iterations)
    print("Final Data.............................")
    print(m1.shape,type(m1))
    print(t.shape,type(t))
    t_temp=t.values
    y_pred=[]
    for i in range(0,13):
        temp_data=(m1[0:13]*t_temp[i])
        temp_data +=m1[13]
        y_pred.append(temp_data)
        temp_data=0
    # np.savetxt('G:\Development\Projects\Python Projects\Machine_Learning\Gradient_Descent\Boston_Project\sub.csv',y_pred, fmt='%0.70f',delimiter="\n")
loading()
