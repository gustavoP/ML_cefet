import pandas as pd
import numpy as np
import static_methods as sm
import matplotlib.pyplot as plt


"""
All vector must be used in 2 dimensions as in np.array([[1, 2, 3]])
Not in 1 dimension as in np.array([1, 2, 3])
"""

data = pd.read_csv('ex1data1.csv', header=None)
plt.figure(0)
data.plot.scatter(x=0,y=1, label='Raw Data')
data.insert(0, 'Ones', 1)
data_array = data.values

x = data_array[:,[0,1]]
y = data_array[:,[2]]


(theta, J)=sm.GD(X=x,y=y,theta =np.zeros((x.shape[1],1)),alfa = 0.001, max_iter=30)
print("The estimated profit of a city with a population of {} is ${:.0f}, and for a population of {} the profit is ${:.0f}" 
    .format(35000,theta[0,0]+theta[1,0]*35000,70000,theta[0,0]+theta[1,0]*70000))
sm.plot(x=x,J=J,theta=theta, y=y)