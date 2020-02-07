import matplotlib.pyplot as plt
import math
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
x=[i for i in range(-100,100,1)]
y=[]
x=np.array(x)
# print(x)
for i in x:
    if i>=0:
        y.append(0)
    else:
        y.append(1)
y=np.array(y)
# print(y)
plt.scatter(x,y)

from sklearn.linear_model import LinearRegression
algo=LinearRegression()
algo.fit(x.reshape(-1,1),y.reshape(-1,1))
m=algo.coef_[0]
c=algo.intercept_
# print(type(m[0]),type(c[0]))
x_line=np.arange(-100,100,1)
y_line=m*x_line+c
plt.plot(x_line,y_line,"r")
print(type(y_line))
y_line_logistic=[]
for i in y_line:
    y_line_logistic.append(1/(1+(math.exp(-1*i))))


# plt.axis([-1000,1000,-1000,1000])
plt.plot(x,y_line_logistic,'g')
plt.show()
