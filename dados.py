import pandas as pd
import numpy as np
import static_methods as sm

"""
All vector must be used in 2 dimensions as in np.array([[1, 2, 3]])
Not in 1 dimension as in np.array([1, 2, 3])
"""

data = pd.read_csv('ex1data1.csv', header=None)

data.plot.scatter(x=0,y=1, label='Raw Data')
data_array = data.values

x = np.concatenate((np.ones((len(data_array[:,[0]]),1)),data_array[:,[0]]),axis=1) #data_array[:,0] returns a 1 dimensional array
y = data_array[:,[1]]

(theta, J)=sm.GD(X=x,y=y,theta =np.array([[0.0],[0.0]]),alfa = 0.001, max_iter=100)

sm.plot(x=x,J=J,theta=theta)