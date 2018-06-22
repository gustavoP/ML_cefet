import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def compute_cost(X, y, theta, h=None):
    """ 
    X is (m,n) vector.
    y is (m,1)
    theta is (n,2) 

    optional:
    h = X@theta, if 'h' not numpy.ndarray it won't be used

    Return scalar
    """ 

    m = float(len(y)) # ins this case len(y) = len(y[:[0]])

    if (h is not None) and isinstance(h,np.ndarray):
        if h.shape != y.shape :
            raise TypeError("'h' has to be the same dimension as 'y' ")
        else:
            a = h-y 
            
    else:
        a = X@theta-y 
    
    j = 1/(2*m)*(a.T@a)
    return j[0,0]

def GD(X, y, theta, alfa, max_iter=10, tolerance=0.01, use_compute_cost = True):
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

def plot(theta, J, x,y):
    """
    theta is the line parameters
    J is the cost funcion list
    X = input data, with ones column
    """
    plot_x = np.sort(x[:,1])
    plot_y = theta[0,0]+theta[1,0]*plot_x
    
    plt.figure(0)
    plt.plot(list(range(0,len(J))), J,'-', label=r'J($\theta$)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend(loc='best')


    plt.figure(1)
    plt.plot(plot_x, plot_y , '-', label='Line fit', linewidth=2, markersize=12, color='#FF0000')
    plt.xlabel(r'Population of city in $10^4$')
    plt.ylabel(r'Profit in $\$10^4$')
    plt.legend(loc='best')


    if theta.shape[0]==2:
        plt.figure(2)
        t0 = np.arange(-10,10,0.01)
        t1 = np.arange(-1,4,0.01)
        t0_m,t1_m = np.meshgrid(t0,t1)

        J1=np.zeros((len(t0),len(t1)))
        for i in range(len(t0)):
            for j in range(len(t1)):
                t = np.array([[t0[i]],[t1[j]]])
                J1[i][j] = compute_cost(x,y,t)
        
        plt.contour(t0_m,t1_m,J1.T,levels=np.logspace(-1,4,20))
        plt.title("Curvas de nível da função custo")
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_0$')


    plt.show()

def normalize(X):
    for i in range(X.shape[1]):
        std = np.std(X[:,[i]])
        #if std == 0: std=1
        X[:,[i]] = (X[:,[i]] - np.average(X[:,[i]]))/std
    return X

def mapFeature(X, degree):
    """
    X has to have only the features, this excludes ones column.
    Return X polynomial with ones column
    """
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(X)

    