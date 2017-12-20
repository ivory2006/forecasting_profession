from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv('cleantwitter200kdata3.csv')

samples=df.loc[:10000:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:10000,'Profession']

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(3)
samples = poly.fit_transform(samples)
print("finish loading data")

#cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(verbose=True)
lg.fit(samples,target)

test_samples=df.loc[8000:10000:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
test_samples = poly.fit_transform(test_samples)
test_target=df.loc[8000:10000,'Profession']
res = lg.predict(test_samples)
print(res)


Logistic_mean=accuracy_score(test_target,res)
print(Logistic_mean)



