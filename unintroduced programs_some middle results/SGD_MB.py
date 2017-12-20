#-200Kdata,logistic regression,Author:ssb--
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
df=pd.read_csv('cleantwitter200kdata3.csv')
samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
samples_train,samples_test,target_train,target_test = train_test_split(samples,target,test_size=0.2,random_state=0)
classifier = SGDClassifier(loss='log',penalty='l1',alpha=0.001)  # logistic
classifier.partial_fit(samples_train,target_train,classes=target)  # training dataset
target_pred=classifier.predict(samples_test) #testing dataset
accuracy=accuracy_score(target_test,target_pred) #accuracy rate
print ('the accuracy score is',accuracy,'\n')

classifier_l2 = SGDClassifier(loss='log',penalty='l2',alpha=0.001)  # logistic
classifier_l2.partial_fit(samples_train,target_train,classes=target)  # training dataset
target_pred_l2=classifier_l2.predict(samples_test) #testing dataset
accuracy_l2=accuracy_score(target_test,target_pred_l2) #accuracy rate
print ('the accuracy score is',accuracy_l2,'\n')


from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
classifier_l1_onevsrest = OneVsRestClassifier(SGDClassifier(loss='log',penalty='l1',alpha=0.001))  # logistic
classifier_l1_onevsrest.partial_fit(samples_train,target_train,classes=target)  # training dataset
target_pred_l1_onevsrest=classifier_l1_onevsrest.predict(samples_test) #testing dataset
accuracy_l1_onevsrest=accuracy_score(target_test,target_pred_l1_onevsrest) #accuracy rate
print ('the accuracy score is',accuracy_l1_onevsrest,'\n')

from sklearn.preprocessing import PolynomialFeatures
samples_poly=PolynomialFeatures(5)
poly=samples_poly.fit_transform(samples)
samples_train,samples_test,target_train,target_test = train_test_split(poly,target,test_size=0.2,random_state=0)
classifier_poly = SGDClassifier(loss='log',penalty='l1',alpha=0.001)  # logistic
classifier_poly.partial_fit(samples_train,target_train,classes=target)  # training dataset
target_pred_poly=classifier_poly.predict(samples_test) #testing dataset
accuracy_poly=accuracy_score(target_test,target_pred_poly) #accuracy rate
print ('the accuracy score is',accuracy_poly,'\n')

