import numpy as np
import pdb
from sklearn.neural_network import MLPRegressor
count =0
fname='Twod1.tra'
ins=[]
outs=[]
with open(fname) as f:
    for line in f:
        line=line.split(' ')
        if (count % 2 ):
            outs.append(line)
        else:
            ins.append(line)
        count = count + 1
ins=np.array(ins,dtype=float)
outs=np.array(outs,dtype=float)
print '%d lines in file' % count
print ins.shape,outs.shape
#print ins[0]
meanx=np.mean(ins,axis=0)
ins=ins-meanx
print np.mean(outs,axis=0)

clf = MLPRegressor(hidden_layer_sizes=(10,10,10), max_iter=750,solver='lbfgs',activation='tanh',epsilon=1e-08)
clf.fit(ins,outs)
print clf.score(ins,outs)

