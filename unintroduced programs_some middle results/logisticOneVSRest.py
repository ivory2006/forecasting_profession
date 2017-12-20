from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
df=pd.read_csv('dataSample1.csv')
samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']

cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(samples,target,test_size=0.2,random_state=0)
model = OneVsRestClassifier(LogisticRegression()) 
model.fit(X_train, y_train)
predicted = model.predict(X_test)
report=classification_report(y_test,predicted)
print(report)
