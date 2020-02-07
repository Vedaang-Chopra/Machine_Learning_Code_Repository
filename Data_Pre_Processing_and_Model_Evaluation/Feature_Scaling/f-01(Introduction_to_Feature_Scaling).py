# Feature Scaling:- It is part of the pre-processing phase of Machine Learning/Deep Learning.
# It is done to ensure that all the features fall in same range.

# Note:- It is not necessary to scale the dummy variable(features with value 0 and 1 only). Scaling can be done/ but when not done it does't break the model.

#  Practice with datasets and see for yourself!!!!!!!!!!!!!!!!!!

# Also in regression problems sometimes predictions need to be scaled also. Check the regression section for that.!!!!!!

import numpy as np
# Contains all the scaling functions.........
from sklearn import preprocessing

# Chosen data points for demo practice
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])

# By using the scale function, we do standard scaling. Tha means with this scaling the mean is 0 and variance is 1.
X_scaled = preprocessing.scale(X)
print(X_scaled)             # New Scaled datapoints

# prints the mean value
print(X_scaled.mean(axis = 0))  # axis=0 means along the rows , and axis=1 means along the column
# Prints the standard deviation
print(X_scaled.std(axis= 0))

# Note:- Always do feature scaling over the entire dataset, not separately on training data  and testing data. As if we do so
#       The training data and testing data would both have different scaling parameters, which could be disasterous.
# In real life we have get testing data in realtime. So we have to save our scaling parameters of training data,
# and change/scale the testing data according to these parameters.
# Using the Standard Scaler Function we can first fit according the training data and then transform the testing data over the same parameters.
# Here by default the mean is 0, but can be Changed.
scaler = preprocessing.StandardScaler()     # Returns a scaler object which could be used to transform the testing data.
# Just Learns and finds the parameters for scaling. Doesn't scale the data
scaler.fit(X)
# Note: We would have to transform/do scaling for training data again after fitting. So tranform will be called on training data also.
# We could use fit_tranform function to simultaneously learn and fit the training dataset
scaler.fit_transform(X)

# Used to transform any data according to learned parameters
scaler.transform(X)

# Demo:- Data for Testing
X_test = [[1,1,0]]
scaler.transform(X_test)

# Similar to Standard Scaler, we have Min-Max Scaler/ Max Absolute Scaler.