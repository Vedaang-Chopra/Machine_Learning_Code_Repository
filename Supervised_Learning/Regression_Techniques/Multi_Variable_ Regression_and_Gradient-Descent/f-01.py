import numpy as np

def cost_calc(x_train,y_train,m,c):
    cost=0
    cost=(y_train-((x_train*m)+c))**2
    # for i in range(0,len(x_train)):
    #     cost=cost+((1/len(x_train))*(y_train[i]-((m*x_train)+c))**2)
    cost=cost/len(x_train)
    return cost.sum()


def step_grad(x_train,y_train,learning_rate,m,c):
    m_slope = 0
    c_slope = 0
    # for i in range(0,len(x_train)):
    #     m_slope = m_slope - ((-2 / (len(x_train))) * (y_train[i]-m*x_train[i]-c))*x_train[i]
    #     c_slope = c_slope - ((-2 / (len(x_train))) * (y_train[i] - m * x_train[i] - c))
    m_slope=(-2/len(x_train))*((y_train-((m*x_train)+c))*x_train)
    c_slope = (-2 / len(x_train)) * (y_train - ((m * x_train) + c))
    new_m = m - learning_rate * m_slope.sum()
    new_c = c - learning_rate * c_slope.sum()
    return new_m , new_c


def grad_descent(x_train,y_train,learning_rate,num_iterations):
    m=0
    c=0
    prev_cost=0
    cost=0
    for i in range(num_iterations):
        m,c=step_grad(x_train,y_train,learning_rate,m,c)
        prev_cost = cost
        cost = cost_calc(x_train, y_train, m, c)
        print("The cost for m:",m," and c:",c," is:",cost)
    return m,c,cost


def loading():
    data=np.loadtxt('data.csv',delimiter=",")
    # print(data)
    print(data.shape)
    x=data[:,0]
    x=x.reshape(-1,1)
    y=data[:,1]
    y=y.reshape(-1,1)
    print(x.shape,y.shape)
    from sklearn import model_selection
    x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=1)
    learning_rate=0.0001
    num_iterations=10
    print(type(x_train))
    m,c,cost=grad_descent(x_train,y_train,learning_rate,num_iterations)
    print("Testing Data.............................")
    m, c ,cost= grad_descent(x_test, y_test, learning_rate, num_iterations)


loading()