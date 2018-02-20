import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sys
from LoadMNIST import load_mnist
def get_data(split_percentage):
    X_images,X_labels=load_mnist("training",np.arange(10),".")
    X_labels=np.array(X_labels)
    #X_images,X_labels=load_mnist("testing",np.arange(10),".")
    print X_labels.shape,X_images.shape
    print X_labels[30]
    X_images=np.reshape(X_images,(60000,784))
    X_images=np.reshape(X_images,(60000,784))
    print X_labels.shape,X_images.shape
    test_image=np.reshape(X_images[30],(28,28))
    plt.imshow(test_image,cmap='gray')
    plt.show()
    #mnist = fetch_mldata("MNIST original")
    #X, y = mnist.data / 255., mnist.target
    #num_train_rows=split_percentage*70000/100
    #num_test_rows=70000-num_train_rows
    #X_train, X_test = X[:num_train_rows], X[num_train_rows:]
    #y_train, y_test = y[:num_train_rows], y[num_train_rows:]
    #return X_train,X_test,y_train,y_test 
    return [],[],[],[]
def split_it_up(train_percentage):
    print "in routine split_it_up",train_percentage
    return 1

def classify(X_train,X_test,y_train,y_test):
# rescale the data, use the traditional train/test split
    mlp = MLPClassifier(hidden_layer_sizes=(25,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
    mlp.fit(X_train, y_train)
    #random_index_match=1423
    #random_index_no_match=1422
    print("Training set score: %f" % mlp.score(X_train, y_train)) #for project dump
    print("Test set score: %f" % mlp.score(X_test, y_test)) #to file if convenient
    #match_image=np.reshape(X_test[random_index_match],(28,28))
    #plt.imshow(match_image,cmap='gray')
    #plt.savefig('sample.png')

if __name__=="__main__":
    p=int(sys.argv[1])
    Xtr,Xte,ytr,yte=get_data(p)
    #classify(Xtr,Xte,ytr,yte)
