import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

x = np.array([i*np.pi/180 for i in range(60, 300, 4)])
y = np.sin(x) + np.random.normal(0, 0.15, len(x))

#data = pd.DataFrame(np.column_stack([x,y]), columns=['x', 'y'])
#plt.plot(data['x'], data['y'], '.')

plt.plot(x, y, '.')
plt.show()


#Initialize a dataframe to store the results:
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define the powers for which a plot is required:
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}

#Iterate through all powers and assimilate results
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
