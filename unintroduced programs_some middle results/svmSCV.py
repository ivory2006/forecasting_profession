#-SVM,Author:ssb--
from numpy import * 
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics, cross_validation
from sklearn import datasets
import pandas as pd
df=pd.read_csv('dataSampleNumber.csv')
samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
print(cv_folds)
for train_index,test_index in cv_folds:
    print(train_index,test_index)
score = cross_validation.cross_val_score(SVC(), samples, target, cv=cv_folds)
print("The scores of cross validation are",score,"respectively")
print("Accuracy is",(score.mean()))
print("The standard deviation is",(score.std()))


