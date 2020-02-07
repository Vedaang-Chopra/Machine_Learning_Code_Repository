# This is a basic demo code of KNN Algorithm ....................................

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier          # Importing KNN Classifier
from sklearn.model_selection import train_test_split

# Loading the Breast Cancer Dataset..................
dataset = datasets.load_breast_cancer()
# print(dataset.target)

# Splitting the Dataset.............
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state = 0)

# Using a KNN Classifier and understanding parameters..............
# n_neighbours: Specify how many numbers of data points in vicinity are to be checked. Default value=5
# metric: Specify which type of distance metric is to be used. Default is minkowski distance. We can also pass our own distance function.
#           Also the minkoski distance is given as (|xi1-xi2|)^p. If p=2 then it is euclidean in nature.
# weights: Specify what kind of weights do we specify on nearest neighbours selected.
#          Can be uniform(No focus on weights, every class has uniform weights), 'distance'(inversely proptional to distance), or a user defined function.
# algorithm: The algorithm used to select nearest neighbours. Default value is auto( best metric is selected automatically).
#            Other Options are : ball_tree, kd_tree, brute also.
# metric_params: If we want to use our own metric function, use this.
# n_jobs: If we wish to do parallel processing while selecting neighbours specify it to 1.
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))


