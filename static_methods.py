import numpy as np

def compute_cost(X, y, theta):
    """ 
    X is (m,n) vector.
    y is (m,1)
    theta is (n,1) 

    Return scalar
    """
    #h = theta'*x
    #j = 1/m sum((h(i)-y(i))^2 )
    #a=X*theta-y;
    #J = 1/(2*m)*(a)'*a;
    
    a = X@theta-y 
    j = 1/(2*len(y))*(a.T@a)
    return j[0,0]
