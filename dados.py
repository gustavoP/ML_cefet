import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import static_methods as sm

a = pd.read_csv('ex1data1.csv', header=None)
a.plot.scatter(x=0,y=1)
plt.show()
