

# -*- coding: utf-8 -*-  
    
from sklearn.cluster import Birch    
from sklearn.cluster import KMeans    
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv('dataSample1.csv')
samples=df.loc[:,['Profession','Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
  
# Kmeans聚类  
clf = KMeans(n_clusters=7)    
y_pred = clf.fit_predict(samples)    
for i  in y_pred:
    print(i)
#print(y_pred)    
  

