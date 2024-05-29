# iris_knn.py

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Iris dataset
dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Initialize and train the k-nearest neighbors classifier
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

# Make predictions on the test set and print results
for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    print("TARGET=", y_test[i], dataset["target_names"][y_test[i]], "PREDICTED=", prediction, dataset["target_names"][prediction])

# Print the accuracy of the classifier
print(kn.score(X_test, y_test))
