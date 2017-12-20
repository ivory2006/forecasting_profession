#-minibatch,Author:ssb--
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

cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
from sklearn.linear_model import LogisticRegression,SGDClassifier
Logistic_predict = cross_validation.cross_val_predict(OneVsRestClassifier(SGDClassifier(loss='log',penalty='l2', alpha=0.001)), samples, target, cv=cv_folds)
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

from sklearn.neural_network import MLPClassifier
Logistic_predict_nn = cross_validation.cross_val_predict(MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1), samples, target, cv=cv_folds)
Logistic_mean_nn=accuracy_score(target,Logistic_predict_nn)
report_nn=classification_report(target,Logistic_predict_nn) #report
lines_nn = report_nn.split('\n')
#pprint(lines)
NewLines_nn=lines_nn[15].split(' ') 
#pprint(NewLines)
Logistic_precision_nn=float(NewLines_nn[9])
Logistic_recall_nn=float(NewLines_nn[15])
Logistic_f1score_nn=float(NewLines_nn[21])
print("Logistic accuracy mean is",Logistic_mean_nn)
print("the Logistic precision is",Logistic_precision_nn)
print("the Logistic recall is",Logistic_recall_nn)
print("the Logistic f1score is",Logistic_f1score_nn,'\n')
#PCA
from sklearn.decomposition import PCA
pca= PCA(n_components=10).fit(samples)
print ('PCA:',pca.explained_variance_ratio_,'\n')
#feature importance ranking
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier() # build extra tree model
model.fit(samples,target)
print('feature importance ranking:',model.feature_importances_,'\n') #display importance of each variables
#Recursive Feature Elimination
from sklearn.feature_selection import RFE
LR = LogisticRegression() # build logistic regression model
rfe = RFE(LR,5) # limit number of variables to three
rfe = rfe.fit(samples,target)
print('Recursive Feature Elimination support',rfe.support_) 
print('Recursive Feature Elimination ranking',rfe.ranking_)

#pca = PCA(n_components=0.95)
#pca.fit(samples)
#print (pca.explained_variance_ratio_)
#print (pca.explained_variance_)
#print (pca.n_components_)


