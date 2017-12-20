#-logistic,Author:ssb--
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv('dataSample1.csv')
samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
from sklearn.linear_model import LogisticRegression
Logistic_predict = cross_validation.cross_val_predict(LogisticRegression(), samples, target, cv=cv_folds)
for (fold_no in 1:cv_folds):
    report=classification_report(target,Logistic_predict) #report
    print(report)



