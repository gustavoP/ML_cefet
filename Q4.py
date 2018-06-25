import numpy as np
import pandas as pd
import static_methods as sm
import static_methods_Logistic as sml
import matplotlib.pyplot as plt
import scipy.optimize as opt 


def plot_logistic(theta,degree,lam=0):
    t1 = np.arange(-0.8,1.2,0.01)
    t2 = np.arange(-0.8,1.2,0.01)
    t1_m,t2_m = np.meshgrid(t1,t2)

    h=np.zeros((len(t1),len(t2)))
    for i in range(len(t1)):
        for j in range(len(t2)):
            t = np.array([[t1[i], t2[j]]])
            h[i][j] = sm.mapFeature(t,degree)@theta
    
    plt.contour(t1_m,t2_m,h.T,levels=[0])
    plt.title(r'Dados dos testes e curva de contorno para $\lambda$ = {}'.format(lam))
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")
    plt.ylim(np.min(t1),np.max(t1))
    plt.xlim(np.min(t2),np.max(t2))
    plt.show()  


"""
All vector must be used in 2 dimensions as in np.array([[1, 2, 3]])
Not in 1 dimension as in np.array([1, 2, 3])
"""

data = pd.read_csv('ex2data2.csv', header=None, names=['test1', 'test2', 'accepted'])  

good = data[data['accepted']==1]
bad = data[data['accepted']==0]

fig, ax = plt.subplots()  
ax.scatter(good['test1'], good['test2'], s=50, c='g', marker='o', label='Accepted')  
ax.scatter(bad['test1'], bad['test2'], s=50, c='y', marker='x', label='Rejected')  
ax.legend()  
ax.set_xlabel('Test 1')  
ax.set_ylabel('Test 2')  

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
degree=6
X = sm.mapFeature(np.array(X.values),degree)
y = np.array(y.values)  
theta = np.zeros(X.shape[1])  

print("Custo = {:.3f}, para theta = 0".format(sml.compute_cost_logistic(theta,X,y)))

lam=1
result = opt.fmin_tnc(func=sml.compute_cost_logistic, x0=theta, fprime=sml.grad_logistic, args=(X, y,lam), disp=0)  



pred = sml.predict_logistic(np.array([result[0]]),X)
acerto = sml.accuracy(pred=pred, y=y)
print("Prediction rate {:.2%}".format(acerto))

plot_logistic(theta=np.array([result[0]]).T,degree=degree, lam=lam)
