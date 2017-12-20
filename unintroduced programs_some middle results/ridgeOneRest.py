#-logistic,one vs rest Author:ssb--
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

k=1
ridge_accuracy =['','','','','','','','','','','','']
ridge_std =['','','','','','','','','','','','']
Lasso_accuracy =['','','','','','','','','','','','']
Lasso_std =['','','','','','','','','','','','']
for i in range(1,13,1):
    fname="{}.csv".format(k)
    df=pd.read_csv(fname)
    samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
    target=df.loc[:,'Profession']
    cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
    #ridge
    from sklearn.linear_model import Ridge
    ridge_predict = cross_validation.cross_val_predict(Ridge(), samples, target, cv=cv_folds)
    ridge_score =cross_validation.cross_val_score(Ridge(), samples, target, cv=cv_folds)
    ridge_accuracy[k-1]=ridge_score.mean()   
    ridge_std[k-1]=ridge_score.std()
    print(ridge_accuracy[k-1])
    print(ridge_std[k-1])
    from sklearn.linear_model import Lasso
    #Lasso
    Lasso_predict = cross_validation.cross_val_predict(Lasso(), samples, target, cv=cv_folds)
    Lasso_score =cross_validation.cross_val_score(Lasso(), samples, target, cv=cv_folds)
    Lasso_accuracy[k-1]=Lasso_score.mean()   
    Lasso_std[k-1]=Lasso_score.std()
    print(Lasso_accuracy[k-1])
    print(Lasso_std[k-1],'\n')
