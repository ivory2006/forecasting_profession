from sklearn import linear_model
from sklearn.linear_model import Ridge,Lasso
from sklearn import metrics, cross_validation
from sklearn import datasets
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
import numpy as np

df=pd.read_csv('cleantwitter200kdata.csv')
samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
#ridge

ridge_predict = cross_validation.cross_val_predict(Ridge(), samples, target, cv=cv_folds)
ridge_score =cross_validation.cross_val_score(Ridge(), samples, target, cv=cv_folds)
ridge_accuracy=ridge_score.mean()   
ridge_std=ridge_score.std()
print(ridge_accuracy)
print(ridge_std)
#Lasso

Lasso_predict = cross_validation.cross_val_predict(Lasso(), samples, target, cv=cv_folds)
Lasso_score =cross_validation.cross_val_score(Lasso(), samples, target, cv=cv_folds)
Lasso_accuracy=Lasso_score.mean()   
Lasso_std=Lasso_score.std()
print(Lasso_accuracy)
print(Lasso_std)



