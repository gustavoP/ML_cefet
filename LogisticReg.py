import numpy as np
import pandas as pd
import static_methods as sm
import static_methods_Logistic as sml
import matplotlib.pyplot as plt
import scipy.optimize as opt 

"""
All vector must be used in 2 dimensions as in np.array([[1, 2, 3]])
Not in 1 dimension as in np.array([1, 2, 3])
"""

data = pd.read_csv('ex2data1.csv', header=None, names=['Exam1', 'Exam2', 'Passed'])  

good = data[data['Passed']==1]
bad = data[data['Passed']==0]

fig, ax = plt.subplots()  
ax.scatter(good['Exam1'], good['Exam2'], s=50, c='g', marker='o', label='Admitted')  
ax.scatter(bad['Exam1'], bad['Exam2'], s=50, c='y', marker='x', label='Not Admitted')  
ax.legend()  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')  

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)  
X = sm.mapFeature(X,6)
y = np.array(y.values)  
theta = np.zeros((X.shape[1],1))
 

#print(sml.compute_cost_logistic(theta,X,y))

result = opt.fmin_tnc(func=sml.compute_cost_logistic, x0=theta, fprime=sml.grad_logistic, args=(X, y, 0))  

# Test case 45 ans 85
#z=np.array([[1,45,85]])@ np.array([result[0]]).T
#print("If you take 45 in Exam 1 and 85 in Exam 2 you have {:.2%} chance to be admitted"
#    .format( sml.sigmoid(z[0,0]) )) 

pred = sml.predict_logistic(np.array([result[0]]),X)
acerto = sml.accuracy(pred=pred, y=y)
print("Prediction rate {:.2%}".format(acerto))

sml.plot_logistic(theta=np.array([result[0]]).T)

plt.ylim(np.min(X[:,1]),np.max(X[:,1]))
plt.xlim(np.min(X[:,2]),np.max(X[:,2]))
plt.show()


#print(sm.compute_cost_logistic(X,y=y,theta = initial_theta))

