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

data, std, avg = sm.normalize_features(data)
data.insert(0, 'Ones', 1)
data_array = data.values

x = data_array[:,[0,1,2]]
y = data_array[:,[3]]

(theta, J)=sm.GD(X=x,y=y,theta =np.zeros((x.shape[1],1)),alfa = 0.01, max_iter=30)
#sm.plot(x=x,J=J,theta=theta, y=y)


#cheking predition
size = 2000
dorms = 2
pred = np.array([[1,size,dorms]])

pred[0,1] = (pred[0,1]-avg[0])/std[0]
pred[0,2] = (pred[0,2]-avg[1])/std[1]

y = (pred@theta)*std[2]+avg[2]
print("For a {} sqf house with {} dorms, the price would be around {:.2f}$".format(size, dorms, y[0][0]))
