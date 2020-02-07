# Using the Grid Search Technique to find the optimum hyper-parameters for the algortihm
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
# Using Grid Search to find the optimum parameter.
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Loading the Iris Dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Splitting the IRIS Dataset
x_train,x_test, y_train, y_test = train_test_split(x, y)

# Find the best hyper parameters, say gamma and c for an rbf kernel can be crucial in increasing the accuracy.
# But brute-forcing the testing dataset for multiple c and gamma is wrong as :
# 1. We are essentially using the testing data for training our algorithm
# 2. Highly Time Consuming
# To avoid this we use cross-validation technique(splitting training data into equal part,
# leaving one partition for hyper parameter optimization and rest for training data).

# Using Grid search technique to find the best possible c and gamma values for SVM algortihm
# Grid search Technique:- It is a brute-force technique which is trying all possible of say all parameters given and find the optimal solution.

print(' Grid Search on SVM algorithm and IRIS Dataset....................')
# Consider the SVM classifier
clf = svm.SVC()
# Here we will brute force on all C and gamma values that are given to it and find the best score within all these combinations.
# Create a dictionary with the parameter of classifier as key and value for it is the different values we want.
grid = {'C' : [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
       'gamma' : [1e-3, 5e-4, 1e-4, 5e-3]}
# The GridSearch takes the classifer and dictionary that holds the values of all parameters which need to be tried.
abc = GridSearchCV(clf, grid)
abc.fit(x_train, y_train)
# The Best estimator gives the best values of hyper-parameter that gives the most score value
print(abc.best_estimator_)

# This shows all the brute forced combinations score, cross-validation dataset etc. Shows Detailed information.
print(abc.cv_results_)

print(' Grid Search on KNN algorithm and IRIS Dataset....................')
# Using Grid search technique to find the best possible neighbours value for KNN algortihm
clf = KNeighborsClassifier()
# using Grid Search on KNN to find the best possible neigbours value.
grid = {"n_neighbors":[3,5,7,9,11]}
abc = GridSearchCV(clf, grid)
abc.fit(x_train, y_train)
# Shows the best neighbours value for the dataset.
print(abc.best_estimator_)

