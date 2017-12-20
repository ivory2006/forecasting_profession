from pandas import Series,DataFrame
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
df = DataFrame(abs(randn(10,5)),columns=['A','B','C','D','E'],index = np.arange(0,100,10))
df.plot(kind='bar')
plt.show()
