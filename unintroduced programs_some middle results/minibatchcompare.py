#-Author:ssb--
from numpy import * 
from sklearn.datasets import load_iris     # import datasets
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score

df=pd.read_csv('dataSample1.csv')

samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
samples_train,samples_test,target_train,target_test = train_test_split(samples,target,test_size=0.3,random_state=0)


# import the LogisticRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score
classifier = LogisticRegression()  # logistic
classifier.fit(samples_train,target_train)  # training dataset
target_pred=classifier.predict(samples_test) #testing dataset
accuracy=accuracy_score(target_test,target_pred) #accuracy rate
matrix=confusion_matrix(target_test,target_pred) #matrix
report=classification_report(target_test,target_pred) #report

print ('the accuracy score is',accuracy,'\n')
print ('the matrix is\n',matrix,'\n')
print ('the classification report is\n',report)
