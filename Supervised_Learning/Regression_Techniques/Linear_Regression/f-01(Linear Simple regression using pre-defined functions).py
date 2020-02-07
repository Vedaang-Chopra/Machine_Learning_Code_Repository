# Linear Regression:
# Here we test the linear regression method by working on a dummy data.
# By calculating the best fit line, the error etc. we check the linear regression algorithm.
import numpy as np
data=np.loadtxt('data.csv',delimiter=",")
# The loadtxt function loads a data set from a text file. The delimiter are a specific set of characters used to differentiate
# between the elements of a row. It takes file path as input and by default the delimiter is space.
print(data.shape)
# print(data)
# The data is used for a simple regression. So there is only one x(feature) and output(y).
# Using -1 allows to calculate the appropriate value for no of rows, if we specify the no. of columns
x=data[:,0].reshape(-1,1)                     # Separating the input and output
y=data[:,1].reshape(-1,1)                     # Both of these are np arrays
print(x.shape)
# print(x)
# The reshape function is used to change the shape.(Explanation as to why we change shape is given below.)
from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)     # Splitting into training and testing data. 75/25 split default.
from sklearn.linear_model import LinearRegression
algo=LinearRegression()
# Algo is an object of Linear regression.
algo.fit(x_train,y_train)
# Note:- The fit function takes only 2-d arrays(because mostly there is more than one feature).
# If we put a 1-d array there would be an error.
# Thus as here x_train and y_train were 1-d arrays, therefore
# we change these 1-d arrays into 2-d arrays using the reshape function.
# Note:- After  fitting the algorithm we have obtained the best fit line. Thus to see the best fit line we plot it.
m=algo.coef_[0]                        # The m(slope) parameter of the best fit line. It returns an array hence the [0].
c=algo.intercept_                      # The c(y-intercept) parameter of the best fit line.
print(m,c)
# y_pred=algo.predict(x_test)
# print(y_pred)

# Comparing The predicted and actual values........................................
# Method-1:Plotting Graphs:
# Now we have the best fit line. So we plot are training data and best fit line.
import matplotlib.pyplot as plt
x_line=np.arange(30,80,0.1)             # An np array with data from 0 to 10.
y_line=m*x_line+c                       # An np array with the y-coordinates of the best fit line according to x_line data.
train_x=x_train.reshape(75)             # Converting to 1-d array to provide input to scatter function as it requires 1-d arrays.
train_y=y_train.reshape(75)
test_x=x_test.reshape(25)
test_y=y_test.reshape(25)
plt.plot(x_line,y_line,"r")             # Checking the training data.
plt.scatter(train_x,train_y)
plt.show()
plt.plot(x_line,y_line,"r")             # Checking the testing data
plt.scatter(test_x,test_y)
plt.show()
# Note:- After checking on testing data, we will not be able to predict the actual values and
#        some values will also have large errors, but some of them will also be correct. Machine learning will
#        not result in correct predictions always.
# Method 2:-Coefficient of Determination:
# We calculate the score of the predicted values.
score_train=algo.score(x_train,y_train)
# The score function takes two parameters the x and y arrays.
score_test=algo.score(x_test,y_test)
print(score_train,score_test)
# Note:-With every run of the file the score will change as the splitting of data is random.
#       Thus on every run data is different and so is the score.
# Whether the score value is good or bad depends on the data set.