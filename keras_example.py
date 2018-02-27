from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data", delimiter=",")
# split into input (X) and output (Y) variables
initial_split=0.7
X = dataset[:,0:8]
Y = dataset[:,8]
num_rows=len(Y)
split_index=int(initial_split*num_rows)
Xtr=X[:split_index,:]
Xte=X[split_index:,:]
ytr=Y[:split_index]
yte=Y[split_index:]
# create model
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(Xtr, ytr, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(Xte, yte)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

