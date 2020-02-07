# Building a python implementation of KNN Classifier....................

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier          # Importing KNN Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter


# Loading the Breast Cancer Dataset..................
dataset = datasets.load_breast_cancer()
# print(dataset.target)

# Splitting the Dataset.............
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state = 0)

k=3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, Y_train)
print('Score with SKLearn implementaion with k=',k,' is:', clf.score(X_test, Y_test))


# Creating own Classifier Functions:
def train(x, y):
    return

def predict_one(x_train, y_train, x_test, k):
    distances = []
    for i in range(len(x_train)):
        distance = ((x_train[i, :] - x_test)**2).sum()
        distances.append([distance, i])
    distances = sorted(distances)
    targets = []
    for i in range(k):
        index_of_training_data = distances[i][1]
        targets.append(y_train[index_of_training_data])
    return Counter(targets).most_common(1)[0][0]

def predict(x_train, y_train, x_test_data, k):
    predictions = []
    for x_test in x_test_data:
        predictions.append(predict_one(x_train, y_train, x_test, k))
    return predictions

y_pred = predict(X_train, Y_train, X_test, k)
print('Our Classifier Score with k=',k,' is:',accuracy_score(Y_test, y_pred))

