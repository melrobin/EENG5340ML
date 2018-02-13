import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
print mnist.target.shape,mnist.data.shape
X_train, X_test = X[:65000], X[65000:]
y_train, y_test = y[:65000], y[65000:]
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train, y_train)
random_index_match=1423
random_index_no_match=1422
print("Training set score: %f" % mlp.score(X_train, y_train)) #for project dump
print("Test set score: %f" % mlp.score(X_test, y_test)) #to file if convenient
match_image=np.reshape(X_test[random_index_match],(28,28))
plt.imshow(match_image,cmap='gray')
plt.savefig('sample.png')

