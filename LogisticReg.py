import numpy as np
import pandas as pd
import static_methods as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

"""
All vector must be used in 2 dimensions as in np.array([[1, 2, 3]])
Not in 1 dimension as in np.array([1, 2, 3])
"""

data = pd.read_csv('ex2data1.csv', header=None, names=['Exam1', 'Exam2', 'Passed'])  
data.insert(0, 'Ones', 1)

good = data[data['Passed']==1]
bad = data[data['Passed']==0]

fig, ax = plt.subplots()  
ax.scatter(good['Exam1'], good['Exam2'], s=50, c='k', marker='+', label='Admitted')  
ax.scatter(bad['Exam1'], bad['Exam2'], s=50, c='y', marker='o', label='Not Admitted')  
ax.legend()  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')  

initial_theta = np.zeros((3,1))

X=data.values[:,[0,1,2]]
y=data.values[:,[3]]

theta, j = sm.GD_logsitoc(X=X,y=y,theta = initial_theta,alfa=0.1)
sm.plot_logistic(X=X,y=y, theta=theta)
plt.show()


#print(sm.compute_cost_logistic(X,y=y,theta = initial_theta))


res = minimize(sm.compute_cost_logistic, initial_theta, args=(X,y), method=None, jac=sm.Grad_logsitoc, options={'maxiter':400})
