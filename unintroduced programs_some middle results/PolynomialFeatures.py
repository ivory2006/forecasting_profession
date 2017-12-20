from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('datasample1.csv')
samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
from sklearn.preprocessing import PolynomialFeatures
samples_poly=PolynomialFeatures(2)
poly=samples_poly.fit_transform(samples)

cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
from sklearn.linear_model import LogisticRegression,SGDClassifier
Logistic_predict = cross_validation.cross_val_predict(OneVsRestClassifier(SGDClassifier(loss='log',penalty="l2", alpha=0.001)), poly, target, cv=cv_folds)
Logistic_mean=accuracy_score(target,Logistic_predict)
report=classification_report(target,Logistic_predict) #report
lines = report.split('\n')
#pprint(lines)
NewLines=lines[15].split(' ') 
#pprint(NewLines)
Logistic_precision=float(NewLines[9])
Logistic_recall=float(NewLines[15])
Logistic_f1score=float(NewLines[21])
print("Logistic accuracy mean is",Logistic_mean)
print("the Logistic precision is",Logistic_precision)
print("the Logistic recall is",Logistic_recall)
print("the Logistic f1score is",Logistic_f1score,'\n')


