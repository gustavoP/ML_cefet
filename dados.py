import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import static_methods as sm

"""
All vector must be used in 2 dimensions as in np.array([[1, 2, 3]])
Not in 1 dimension as in np.array([1, 2, 3])
"""

a = pd.read_csv('ex1data1.csv', header=None)
a.plot.scatter(x=0,y=1)
#plt.show()

#variables for test
x = np.array([[1,2],[1,1],[1,2]])
y = np.array([[0.1],[0.2],[0.4]])
theta = np.array([[1],[0.6]])

print(sm.GD(X=x,y=y,theta =theta,alfa = 0.1))