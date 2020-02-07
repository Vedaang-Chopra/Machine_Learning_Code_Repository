# This code helps to build a Random Forest Classifier...............
# It also analyses the accuracy of Random Forest and Decision Tree on training Data and Testing Data
# Go through documentation of Random Forest for other options to reduce over fitting and optimizing classifier,
from sklearn import model_selection
from sklearn import datasets

# Importing Breast Cancer Dataset......................
breast_cancer=datasets.load_breast_cancer()
# print(breast_cancer)

# Splitting Data into Training and Testing..................
x_train,x_test,y_train,y_test=model_selection.train_test_split(breast_cancer['data'],breast_cancer['target'],random_state=0)

# Importing Random Forest Classifier.....................
from sklearn import ensemble
# random_state: Helps create same random forest trees every time
# max_depth: Reduce over fitting by specifying the max levels of splitting

clf_forest=ensemble.RandomForestClassifier(random_state=0,max_depth=10)
clf_forest.fit(x_train,y_train)
y_pred=clf_forest.predict(x_test)
print('Score of Random Forest on Training Data:',clf_forest.score(x_train,y_train))
print('Score of Random Forest on Testing Data:',clf_forest.score(x_test,y_test))

# Importing Decision Tree Classifier.....................
from sklearn import tree
clf_tree=tree.DecisionTreeClassifier()
clf_tree.fit(x_train,y_train)
y_pred=clf_tree.predict(x_test)
print('Score of Decision Tree on Training Data:',clf_tree.score(x_train,y_train))
print('Score of Decision Tree on Testing Data:',clf_tree.score(x_test,y_test))

# As expected the Training Data accuracy will decrease in random Forest due to random trees and less features,
# but due to multiple trees used together the Testing Data Accuracy increases.
