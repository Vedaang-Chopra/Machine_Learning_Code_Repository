# A simple code to learn the Decision Tre Classifier


from sklearn import datasets
# Importing the Decision Tree Classifier.............
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Importing the export_graphviz to visualize the decision tree that has been created by classifier. It needs to be downloaded additionally into Anaconda.
from sklearn.tree import export_graphviz

# The pydotplus helps us to save the decision tree created into a PDF File. It also needs to be downloaded additionally into the Anaconda interpreter.
import pydotplus as py

# Loading the IRIS Dataset......................
iris=datasets.load_iris()
# Splitting Data into Training and Testing
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=1)

# Creating the Decision Tre Classifier and fitting it onto the iris dataset
clf=DecisionTreeClassifier()
# The fit functions learns on the dataset
clf.fit(x_train,y_train)

# The predict function predicts on the dataset values given to it.
y_pred_train=clf.predict(x_train)
y_pred_test=clf.predict(x_test)


# The Confusion matrix helps us to anlayze how many predictions were made right or wrong.
a=confusion_matrix(y_train,y_pred_train)
b=confusion_matrix(y_test,y_pred_test)
# high Training Dataset Accuracy means model is overfitted. Verify the model by comparing testing data values.
print('Training Data Prdictions:')
print(a)
print('Testing Data Prdictions:')
print(b)


# Visualizing the Decision Tree and Exporting it into PDF
# The export_graphviz helps to visualize the decision tree created. The important Parameters for it are:
# out_file: The export_graphviz returns a dot_data object. The out_file saves this object into a file .pdot.
#           If we don't wish to save it to a file ,specify it to be None else specify some name for the file with in which it has to be saved.
# feature_name: Adds feature_names to the nodes
# class_name: Adds class_names to the nodes
# We can also add other features such as make boxes rounder, use colour etc. Refer Documentation

# Note:- Here i have to fix the error of "GraphViz's executables not found"

dot_data=export_graphviz(clf,out_file=None,feature_names=iris.feature_names,class_names=iris.target_names)
# To convert the dot_data object and save it in a pdf format use the following functions.
graph=py.graph_from_dot_data(dot_data)
graph.write_pdf("Iris_dataset_Decision_Tree.pdf")             # Writes to pdf File
print(graph)

