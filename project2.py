import numpy as np
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense

def get_data(fname):
    ins=[]
    outs=[]
    count =0
    with open(fname) as f:
        for line in f:
            line=line.split()
            if (count % 2 ):
                outs.append(line)
            else:
                ins.append(line)
            count = count + 1
    ins=np.array(ins,dtype=float)
    outs=np.array(outs,dtype=float)
    meanx=np.mean(ins,axis=0)
    stdx=np.std(ins,axis=0)
    ins=(ins-meanx)/stdx
    return ins,outs
def create_mlp_model(hidden_layer=(10,),iters=7500):
    clf = MLPRegressor(hidden_layer_sizes=hidden_layer, max_iter=iters,solver='lbfgs',activation='tanh',epsilon=1e-08)
    return clf

def calculate_mse(outs,pred):
    A=outs-pred
    A=np.dot(A.T,A)
    mse=np.trace(A)/np.size(pred,0)
    return mse

fname_train='Twod1.tra'
fname_test='Twod.tst'
X,y=get_data(fname_train)
Xtest,ytest=get_data(fname_test)
model_train=create_mlp_model(hidden_layer=(10,),iters=750)
model_train.fit(X,y)
pred_train=model_train.predict(X)
pred_val=model_train.predict(Xtest)
train_mse=calculate_mse(y,pred_train)
val_mse=calculate_mse(ytest,pred_val)
model = Sequential() #Start of a keras implementation
model.add(Dense(10, input_dim=8, use_bias=True,activation='tanh'))
model.add(Dense(7))
# Compile model
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])
# Fit the model
model.fit(X,y, epochs=50)
keras_pred_train=model.predict(X, verbose=0, steps=None)
keras_pred_val=model.predict(X, verbose=0, steps=None)
print calculate_mse(y,keras_pred_train) 
score_train=model.evaluate(X,y,verbose=0)[1]
score_val=model.evaluate(Xtest,ytest,verbose=0)[1]
# evaluate the model
print 'Keras results'
print 'Training error: %f\tValidation error: %f'%(score_train,score_val)
print 'Scikit-Learn MLP results'
print 'Training error: %f\tValidation error: %f'%(train_mse,val_mse)
