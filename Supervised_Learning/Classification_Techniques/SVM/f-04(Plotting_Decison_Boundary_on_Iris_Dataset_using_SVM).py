# Using SVM on IRIS_Dataset and doing Multi_class_Classification and Plotting the Information
# Plotting Decision Boundary using SVM...............
# To understand the Plotting Function at the end check tutorial........
# Importing SVM Classifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# Loading the Iris Dataset
iris = datasets.load_iris()
x = iris.data[:,0:2]         # We can use only two features, because to plot on a 2-D plane we need two features only
y = iris.target

# Splitting the Data into Training and Testing
x_train,x_test, y_train, y_test = train_test_split(x, y)

# Creating an SVM Classifier
# While using gaussian kernel we can get non linear boundary.
clf_gaussian = svm.SVC(kernel = 'rbf',C=1)
# Using Linear kernel we get Linear boundary
clf_linear=svm.LinearSVC(C=1)

# Fit function is used to learn on the data
clf_gaussian.fit(x_train, y_train)
clf_linear.fit(x_train, y_train)

# Displays the Accuracy of the Algorithm
print(clf_gaussian.score(x_test, y_test))
print(clf_linear.score(x_test, y_test))


# Important function to Visualise the Output
def makegrid(x1, x2, h = 0.02):
    # Here we are choosing x1 feature and predicting all values in the range [x1.min()-1,x1.max()+1]
    # Here we are choosing x2 feature and predicting all values in the range [x2.min()-1,x2.max()+1]
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    # Create a range for x1,x2 values
    a = np.arange(x1_min,x1_max,h)
    b = np.arange(x2_min, x2_max, h)
    # Creates a grid of a*b that includes all points in the list a and b
    xx, yy = np.meshgrid(a, b)
    return xx, yy

xx, yy = makegrid(x[:, 0], x[:, 1])
# Ravel Function Creates a 2-D array  and flattens it out to 1-D array
predictions_gaussian = clf_gaussian.predict(np.c_[xx.ravel(), yy.ravel()])
predictions_linear = clf_linear.predict(np.c_[xx.ravel(), yy.ravel()])

# Plotting the new values/ Plotting the entire grid
plt.scatter(xx.ravel(), yy.ravel(), c = predictions_gaussian)
plt.show()

plt.scatter(xx.ravel(), yy.ravel(), c = predictions_linear)
plt.show()