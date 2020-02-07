# Using the logistic Regression Classifier...................................

# Importing Classifier and Datasets...........................
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

b_c=datasets.load_breast_cancer()
# print(b_c)
x=b_c.data          # X values
y=b_c.target        # Y Values
# print(x.shape,y.shape)


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)     # Splitting into training and testing data. 75/25 split default.

# Creating a classifier object........
# Refer documentation for hyper-parameter optimization.
lr=LogisticRegression(C=2,solver='saga')
# Some parameter values to take care about:
# 'C': It is the parameter on the original cost instead of the parameters. Importance is on the cost, rather than regularization.
#      It is the parameter similar to regularization parameter but instead it is on the cost function that we calculated.
# Huge C can cause over-fitting and overshadow regularization, very small c can cause under-fitting and is undershadow over regularization.

# 'solver': It is basically used to apply a differnt type of gradient descent a type of optimization algorithm.
#           'lib-linear is slow, sag/saga is faster.
# 'tol': It is applied to check if the change in cost is function is very low, we can set a value after which we gradeint descent could stop.

# 'max_iter': No. of times the gradient iteration has to occur to change the value of m for a better global minima.

# 'multi_class': For a multi-class classification problem we can use this parameter to specify which type of classification.
#               We can use 'ovr' for one vs rest or 'multi' for multinomial method.
# 'penalty': It is the Regularization Parameter, lambda.


# Fit the training data, so that the model can learn the parameters that are required to predict values, find the pattern.
lr.fit(x_train,y_train)

# Predict the values on the the given input.
print(lr.predict(x_test))

# The Score Function tells how correct our predictions are, how well the algorithm is working, what's the mean accuracy of the model
print(lr.score(x_test,y_test))

# To find the mistakes in our model, we are to trying to find those values which are predicted wrong.
# Those values which are -1 or 1 we are making an error because 0-1\1-0.
print(lr.predict(x_test)-y_test)

# We are doing so to to check whether the model was highly wrong or that points or just slightly wrong.
# It tells us about the confidence or parameter value of our models
# To find those values that is the value of h(x(i))) for the class predicted we have the function predict_proba
print(lr.predict_proba(x_test))


# Error on index 3:- was not huge as h(x) value if near about 0.3/0.6 for both the classes.
print(lr.predict_proba(x_test)[3])
