#-Author:ssb--
from numpy import * 
from sklearn.datasets import load_iris     # import datasets
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score

df=pd.read_csv('dataSampleNumber.csv')

samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
samples_train,samples_test,target_train,target_test = train_test_split(samples,target,test_size=0.3,random_state=800)


# import the linearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
classifier = LinearRegression()  # linear
classifier.fit(samples_train,target_train)  # training dataset
target_pred=classifier.predict(samples_test) #testing dataset
explained_variance_score=explained_variance_score(target_test,target_pred) #Explained variance regression score function
mean_absolute_error=mean_absolute_error(target_test,target_pred) #Mean absolute error regression loss
mean_squared_error=mean_squared_error(target_test,target_pred) #Mean squared error regression loss
r2_score=r2_score(target_test,target_pred) #ER^2 (coefficient of determination) regression score function.


print ('the variance regression score is',explained_variance_score,'\n')
print ('Mean absolute error regression loss is',mean_absolute_error,'\n')
print ('Mean squared error regression loss is',mean_squared_error,'\n')
print ('ER^2 (coefficient of determination) regression score',r2_score,'\n')

#x = classifier.predict([0.13,0.69,0.93,0.84,0.40,0.84,0.62,0.64,0.92,0.36])  # input data
#print(x)


