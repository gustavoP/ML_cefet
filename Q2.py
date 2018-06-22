import pandas as pd
import numpy as np
import static_methods as sm
import matplotlib.pyplot as plt


"""
All vector must be used in 2 dimensions as in np.array([[1, 2, 3]])
Not in 1 dimension as in np.array([1, 2, 3])
"""

data = pd.read_csv('ex1data2.csv', header=None)
plt.figure(0)
data.plot.scatter(x=0,y=1, label='Raw Data')

data.insert(0, 'Ones', 1)

data_array = data.values

x = data_array[:,[0,1,2]]
y = data_array[:,[3]]

#xn = sm.normalize(x[:,[1]])

#Falta normalizar
(theta, J)=sm.GD(X=x,y=y,theta =np.zeros((x.shape[1],1)),alfa = 0.0000001, max_iter=30)
sm.plot(x=x,J=J,theta=theta, y=y)