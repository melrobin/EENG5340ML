import numpy as np
import sys
def g(x):
    y=np.zeros((2,1))
    y[0]=2*(x[0]-2)
    y[1]=-np.sin(2*x[1])
    return y
def f(x):
    return (x[0]-2.)*(x[0]-2.)+np.cos(x[1])*np.cos(x[1])

def optim1(x):
    lambda1=0.7
    for i in range(30):
        x=x-lambda1*g(x)
        print "Iteration %d\t%g\t%g"%(i,x[0],x[1])
    return x

def optim2(x):
    lambda1=0.7
    for i in range(30):
        x=x-lambda1*np.linalg.solve(H(x),g(x))
        print "Iteration %d\t%g\t%g"%(i,x[0],x[1])
    return x

def H(x):
    y=np.zeros((2,2))
    y[0,0]=2
    y[0,1]=0
    y[1,0]=0
    y[1,1]=-2*np.cos(2*x[1])
    return y

if __name__=="__main__":
    if (len(sys.argv)<3):
        print "Not enough arguments...dueces!"
        quit()
    x0=np.zeros((2,1))
    x0[0]=float(sys.argv[1])
    x0[1]=float(sys.argv[2])
    sol1=optim1(x0)
    sol2=optim2(x0)
    print "Solution 1 is ",sol1
    print "Value of objective at Solution 1 is %g"%f(x0)
    print "Solution 2 is ",sol2
    print "Value of objective at Solution 2 is %g"%f(x0)

