# This code helps us understand how decision boundaries are built by SVM and how it is affected by change in the 'C' Parameter.............

import numpy as np
# Importing SVM Classifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt



# Case 1:- Understanding the effects of having a small value of C

# Data points to be plotted..............
x=np.array([[1,1],[2,1],[1,2],[1.5,1.5],[3,4],[2,5],[4,3],[7,2],[3,5],[2,6],[6,2],[3,5],[4,4]])
# Classes of plotted Data Points
y=[0,1,0,0,1,1,1,1,1,1,1,1,1]

# Plotting the Data Points on Graph.......................
# print(len(x))
x_x1=x[:,0]
x_x2=x[:,1]
# print(x_x1)
plt.scatter(x_x1,x_x2,c=y)
# plt.show()

# Linear kernel specifies here that there is a linear line to be created for classification
# With a small c we allow error and the line is like a simple SVM classifier, regularization error is not shadowed and is in focus.
svclinear=SVC(kernel='linear',C=1).fit(x,y)
# As the data points here have two features, therefore the linear line has the following line:y=m1*x1+m2*x2+m3
# The coef_ parameter prints the value of m1,m2 that are being calculated.
print('Coefficient of x(m1,m2):',svclinear.coef_)
# The intercept_ parameter calculates the value of the m3 paramter.
print('Constant value(m3):',svclinear.intercept_)

# Now as we found the parameters m1,m2 we are now plotting the line.
# The current function is : y=m1*x1+m2*x2+m3
# To plot a two feature line in 2-D plane we would have to assume one paramter as 0. Say y=0
# We need two points two plot now. We assume f1 to be 0 in one and 5 in the second. We now have too find the corresponding values of f2 to plot line.
x1=np.array([0,5])
# As we have the value of m1,m2 and m3 we would assume value of x1 and y to be 0 to find x2.
# After assuming them to be zero, x2=-m3/m2
# The calculation below is for that:
x2=-1*(svclinear.intercept_+svclinear.coef_[0][0]*x1)/svclinear.coef_[0][1]

# Plotting Line....
plt.plot(x1,x2)
plt.scatter(x_x1,x_x2,c=y)
plt.axis([0,8,0,8])
plt.show()


# Case 2:- Understanding the effects of having a large value of C

# Data points to be plotted..............
x=np.array([[1,1],[2,1],[1,2],[1.5,1.5],[3,4],[2,5],[4,3],[7,2],[3,5],[2,6],[6,2],[3,5],[4,4]])
# Classes of plotted Data Points
y=[0,1,0,0,1,1,1,1,1,1,1,1,1]

# Plotting the Data Points on Graph.......................
# print(len(x))
x_x1=x[:,0]
x_x2=x[:,1]
# print(x_x1)
plt.scatter(x_x1,x_x2,c=y)
# plt.show()

# Linear kernel specifies here that there is a linear line to be created for classification
# With a large value of C the line changes and no error is allowed, and line seems overfitted, with regularization error overshadowed.
svclinear=SVC(kernel='linear',C=10000).fit(x,y)
# As the data points here have two features, therefore the linear line has the following line:y=m1*x1+m2*x2+m3
# The coef_ parameter prints the value of m1,m2 that are being calculated.
print('Coefficient of x(m1,m2):',svclinear.coef_)
# The intercept_ parameter calculates the value of the m3 paramter.
print('Constant value(m3):',svclinear.intercept_)

# Now as we found the parameters m1,m2 we are now plotting the line.
# The current function is : y=m1*x1+m2*x2+m3
# To plot a two feature line in 2-D plane we would have to assume one paramter as 0. Say y=0
# We need two points two plot now. We assume f1 to be 0 in one and 5 in the second. We now have too find the corresponding values of f2 to plot line.
x1=np.array([0,5])
# As we have the value of m1,m2 and m3 we would assume value of x1 and y to be 0 to find x2.
# After assuming them to be zero, x2=-m3/m2
# The calculation below is for that:
x2=-1*(svclinear.intercept_+svclinear.coef_[0][0]*x1)/svclinear.coef_[0][1]

# Plotting Line....
plt.plot(x1,x2)
plt.scatter(x_x1,x_x2,c=y)
plt.axis([0,8,0,8])
plt.show()