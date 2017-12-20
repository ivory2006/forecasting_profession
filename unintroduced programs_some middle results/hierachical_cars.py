import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from pprint import pprint
#from pylab import rcParams
#import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import AgglomerativeClustering
#import sklearn.metrics as sm

np.set_printoptions(precision=4,suppress=True)
#plt.figure(figsize=(100,50))
#%matplotlib inline
plt.style.use('seaborn-whitegrid')

with open("mtcars.csv", "r") as fi: #1,000,000 hashes
    cars = {line.strip() for line in fi}
    pprint(cars)
"""
address='mtcars.csv'
cars=pd.read_csv(address)
cars.columns=['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
x=cars.ix[:,(1,3,4,6)].values
y=cars.ix[:,(9)].values
print(x)
"""




