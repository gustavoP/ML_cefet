import numpy as np
import pandas as pd
import static_methods as sm
import matplotlib.pyplot as plt

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
plt.show()