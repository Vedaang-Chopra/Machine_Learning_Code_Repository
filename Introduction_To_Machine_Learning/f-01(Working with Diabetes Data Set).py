import pandas as pd
import numpy as np
# Sklearn-Scikit Learn
# Here we are going to work with the house pricing data(diab) and work with it.
# Sklearn package already has many data sets, machine learning algorithms and other various modules for machine learning.
from sklearn import datasets
# The "datasets" module has already pre defined data sets within it. They are already cleaned.
diab=datasets.load_diabetes()                       # Loading the diab House Pricing Data set.
print(type(diab))                                 # An Sklearn data set object
# print(diab)
x=diab.data
# The x and y defined are after looking into the data set as the "data" and "target" are defined columns/headers within the data set, and thus are separated exclusively.
y=diab.target
print(type(x),type(y))                              # Both x, y are numpy arrays
print(x.shape,y.shape)
# We convert our data into data frame.
dfx=pd.DataFrame(x)
dfy=pd.DataFrame(y)
print(diab.feature_names)
dfx.columns=diab.feature_names                    # Changing the data frame headers into feature names
print(dfx.describe())
print(diab.DESCR)
# Basically we have to understand the data set by using various functions.
# Till now we have done till step III of supervised learning steps.
# Working with data:-
# Now we randomly choose the data and split it into training and testing.
# If we choose it chronologically then we may miss changing trends that occurred could have occured int later part of the data set.
from sklearn import model_selection
# For selecting the data for training and testing we use model_selection module
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)
# Here the train_test_split function randomly selects the data points for x_train, y_train, x_test, y_test.
# It randomly selects a row and changes the selection on each run of the code and then returns 4 numpy arrays(x_train, y_train, x_test, y_test{order is important}).
print(x_train.shape,type(x_train))
print(y_train.shape,type(y_train))
print(x_test.shape, type(x_test))
print(y_test.shape, type(y_test))
# Now we use machine learning-training algorithms and linear regression to predict output(house costs).
from sklearn.linear_model import LinearRegression
# For learning we need a linear regression algorithm
algo=LinearRegression()
print(type(algo))
algo.fit(x_train, y_train)                                # Now we have our algorithm we train it using the fit function which takes x_train and y_train.
y_pred=algo.predict(x_test)                               # Now we predict the output from our trained algorithm using the predict function.
# We have our predicted output. Now to compare it with y_test(actual output) we plot graph
import matplotlib.pyplot as plt
x_plot=[10,20,30,40,50,60,70,80,90,100]
y_plot=[10,20,30,40,50,60,70,80,90,100]
print(plt.scatter(y_test,y_pred))
plt.show()
plt.plot(x_plot,y_plot)
plt.axis([0,100,0,100])
plt.show()


