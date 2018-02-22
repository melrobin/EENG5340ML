import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sys
from LoadMNIST import load_mnist
def get_data(split_percentage):
    X_images,X_labels=load_mnist("training",np.arange(10),".")
    y_train=np.array(X_labels)
    X_train=np.reshape(X_images,(60000,784))
    X_images,X_labels=load_mnist("testing",np.arange(10),".")
    y_test=np.array(X_labels)
    X_test=np.reshape(X_images,(10000,784))
    print X_train.shape,X_test.shape,y_train.shape,y_test.shape
    X_train= X_train / 255.
    X_test= X_test / 255.
    X=np.vstack((X_train,X_test))
    y=np.concatenate((y_train,y_test))
    #test_image=np.reshape(X_images[30],(28,28))
    #plt.imshow(test_image,cmap='gray')
    #plt.show()
    #mnist = fetch_mldata("MNIST original")
    #num_train_rows=split_percentage*70000/100
    #num_test_rows=70000-num_train_rows
    #X_train, X_test = X[:num_train_rows], X[num_train_rows:]
    #y_train, y_test = y[:num_train_rows], y[num_train_rows:]
    #return X_train,X_test,y_train,y_test 
    return X,y
def split_it_up(train_percentage,X_orig,y_orig):
    num_rows=len(y_orig)
    split_index=train_percentage*num_rows/100
    X_train, X_test = X_orig[:split_index], X_orig[split_index:]
    y_train, y_test = y_orig[:split_index], y_orig[split_index:]
    return X_train,X_test,y_train,y_test 

def classify(X_train,X_test,y_train,y_test):
    print 'Training...'
    #print X_train.shape,X_test.shape,y_test.shape,y_train.shape
# rescale the data, use the traditional train/test split
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1)
    mlp.fit(X_train, y_train)
    the_training_score=mlp.score(X_train, y_train)
    the_validation_score=mlp.score(X_test, y_test)
    print("Training set score: %f" % mlp.score(X_train, y_train)) #for project dump
    print("Test set score: %f" % mlp.score(X_test, y_test)) #to file if convenient
    return the_training_score,the_validation_score
if __name__=="__main__":
    p=int(sys.argv[1])
    fname=sys.argv[2]
    f=open(fname,'a')
    print 'Working with split percentage',p
    X,y=get_data(p)
    Xtr,Xte,ytr,yte=split_it_up(p,X,y)
    train_score,test_score=classify(Xtr,Xte,ytr,yte)
    f.write(str(train_score))
    f.write(',')
    f.write(str(test_score))
    f.write('\n')
