# Using SVM on IRIS_Dataset and doing Multi_class_Classification
# Importing SVM Classifier
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# Loading the Iris Dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target
# Splitting the Data into Training and Testing
x_train,x_test, y_train, y_test = train_test_split(x, y)

# Creating an SVM Classifier and Defining parameters
# kernel:- Used to specify the type of similarity function; Default rbf
# C: parameter to set Decision Boundary; Higher C, training error reduces, can cause overfitting ; Lower C, regularization error reduces, can cause underfitting
# class_weight:- Used to set bias to certain class/ give them prior probability
# coef0:- During Polynomial Kernel the value of a parameter is set through this
# decision_function_shape:- Used to satisfy some API requirements; not necessary; Used to display the output as one vs rest scheme
# degree:- During Polynomial Kernel the value of b parameter is set through this
# gamma:- During Gaussian/rbf kernel the value of entire denominator (one divided by 2 sigma square) is set by this
# For Multi-Class classification SVC uses the one vs one method only.
clf_gaussian = svm.SVC(kernel = 'rbf',C=1)

# Similar to SVC this uses linear kernel as default and one vs rest method for multi class classification.
clf_linear=svm.LinearSVC(C=1)

# Fit function is used to learn on the data
clf_gaussian.fit(x_train, y_train)
clf_linear.fit(x_train, y_train)

# Displays the Accuracy of the Algorithm
print(clf_gaussian.score(x_test, y_test))
print(clf_linear.score(x_test, y_test))

