import numpy as np

def compute_cost(X, y, theta, h=None):
    """ 
    X is (m,n) vector.
    y is (m,1)
    theta is (n,1) 

    optional:
    h = X@theta, if 'h' not numpy.ndarray it won't be used

    Return scalar
    """

    m = float(len(y[0]))

    if (h is not None) and (type(h) is np.ndarray):
        if h.shape != y.shape :
            raise TypeError("'h' has to be the same dimension as 'y' ")
        else:
            a = h-y 
            
    else:
        a = X@theta-y 
    
    j = 1/(2*m)*(a.T@a)
    return j[0,0]

def GD(X, y, theta, alfa, max_iter=10, tolerance=0.1, use_compute_cost = True):
    """
    X is (m,n) vector.
    y is (m,1)
    theta is (n,1) 

    return theta(n,1), J[iter]
    """
    m = float(len(y[0]))
    if use_compute_cost: J = []
     
    for i in range(max_iter):
        h = X@theta
        if use_compute_cost: J.append(compute_cost(X,y,theta,h))
              
        theta -= (alfa/m)*X.T@(h-y)

    if use_compute_cost: return theta,J
    else: return theta
