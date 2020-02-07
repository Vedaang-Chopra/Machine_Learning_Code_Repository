# This Code helps us to understand Cross Validation Technique and Understand Relation between K or n_neighbour and Score...................
# Cross Validation helps us to find the best optimal parameters for a classifer by retaining a part of training data for testing multiple parameter values.

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier          # Importing KNN Classifier
from sklearn.model_selection import train_test_split,KFold  # Used to create a KFold object
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score         # Used to implement Cross Validation

# Loading the Breast Cancer Dataset..................
dataset = datasets.load_iris()
# print(dataset.target)

# Splitting the Dataset.............
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state = 0)

# Using a KNN Classifier to understand Cross Validation..............
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, Y_train)
print('Score of KNN:',clf.score(X_test, Y_test))

# Using Basic Linear Regression to understand Cross Validation............................
clf_linear=LinearRegression()
clf.fit(X_train,Y_train)
print('Score of Linear Regression:', clf.score(X_test,Y_test))

# Using Cross Validation Function and Understanding Parameters.....................
# estimator: Classifier object used to fit the data
# X: Takes the training data
# y: Takes the output of training data
# groups: If we want to specify each and every data point to go into specific training subset manually, then pass it a list.
#         Don't use it, as every data point should go into a random training subset.
# cv: Specify how to create subsets of training data. Use KFold object(splits data into 3 parts)
#     KFold object parameters:- KFold(no of splits to create, (boolean value) whether to shuffle the data or not, random_state value if shuffle is used)
# n_jobs: Parallel processing
# fit_params: If estimators fit function requires some extra parameters, we can pass it through this.

optimum_param=cross_val_score(estimator=clf_linear,X=X_train,y=Y_train,cv=KFold(3,True,0))
# Currently optimum_param holds the scores of 3 different training and testing subsets that created from iris data. Not used to find efficient parameter values
print(optimum_param)

# Here the Score is 0 as iris data holds 3 classes, and when training we didn't provide any datapoint with the 3 class, because shuffle was off and exact 3 subsets were created.
print(cross_val_score(estimator=clf_linear,X=dataset.data,y=dataset.target,cv=KFold(3,False,0)))

# To solve the above problem always make shuffle true in KFold object.
print(cross_val_score(estimator=clf_linear,X=dataset.data,y=dataset.target,cv=KFold(3,True,0)))

# Finding Optimum Parameters with Cross Validation for KNN........................
# Understanding Relation between score and ne_neighbours by Plotting the Score and n_neigbours Parameter

x_axis = []
y_axis = []
# Brute-Forcing to find the optimal K by incrementing the K/n_neighbour value
points=[]
max_score,best_k=0,0
for i in range(1, 26, 2):
    clf = KNeighborsClassifier(n_neighbors = i)
    score = cross_val_score(clf, X_train, Y_train,cv=KFold(3,True,0))
    if max_score <= score.mean():
        max_score = score.mean()
        best_k = i
    # print(i,score.mean())
    x_axis.append(i)
    y_axis.append(score.mean())

# Plotting the Score and K value
print(len(score))
import matplotlib.pyplot as plt
plt.scatter(x_axis,y_axis)
plt.plot(x_axis, y_axis)
plt.show()

print('Best K-Nearest Neighbour Value:',best_k)




