import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, theta, h=None):
    """ 
    X is (m,n) vector.
    y is (m,1)
    theta is (n,1) 

    optional:
    h = X@theta, if 'h' not numpy.ndarray it won't be used

    Return scalar
    """ 

    m = float(len(y)) # ins this case len(y) = len(y[:[0]])

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
    X is (m,n)
    y is (m,1)
    theta is (n,1) 

    return theta(n,1), J[iter]
    """
    m = float(len(y)) # ins this case len(y) = len(y[:[0]])
    if use_compute_cost: J = []
     
    for i in range(max_iter):
        h = X@theta
        if use_compute_cost: J.append(compute_cost(X,y,theta,h))
        
        #check tolerance
        if i>=1:
            if abs(J[-1]-J[-2]) <= tolerance:
                break


        theta -= (alfa/m)*X.T@(h-y)

    if use_compute_cost: return theta,J
    else: return theta

def plot(theta, J, x):
    """
    theta is the line parameters
    J is the cost funcion list
    X = input data, with ones column
    """
    plot_x = np.sort(x[:,1])
    plot_y = theta[0,0]+theta[1,0]*plot_x
    plt.plot(plot_x, plot_y , '-', label='Line fit', linewidth=2, markersize=12, color='#FF0000')

    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.figure(0)

    plt.plot(list(range(0,len(J))), J,'-', label=r'J($\theta$)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend(loc='best')
    plt.figure(1)

    plt.show()