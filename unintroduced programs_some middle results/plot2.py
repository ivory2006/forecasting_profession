import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
#df = pd.read_csv('testData.csv')

df=pd.DataFrame([[40929,4066443],[93904,9611272],[188349,19360005]],index=['A','B','C'],columns=['amount','price'])
fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

df.amount.plot(kind='bar', color='red', ax=ax, width=width, position=1)
df.price.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Amount')
ax2.set_ylabel('Price')
plt.show()
