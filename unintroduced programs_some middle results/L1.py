#-logistic,one vs rest Author:ssb--
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
k=1
Logistic_mean =['','','','','','','','','','','','']
Logistic_std =['','','','','','','','','','','','']
Logistic_precision =['','','','','','','','','','','','']
Logistic_recall =['','','','','','','','','','','','']
Logistic_f1score =['','','','','','','','','','','','']
for i in range(1,13,1):
    fname="{}.csv".format(k)
    df=pd.read_csv(fname)
    samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
    target=df.loc[:,'Profession']
    cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
    #logistic
    from sklearn.linear_model import LogisticRegression
    Logistic_predict = cross_validation.cross_val_predict(LogisticRegression( penalty='l1', tol=0.01), samples, target, cv=cv_folds)
    Logistic_score =cross_validation.cross_val_score(LogisticRegression( penalty='l1', tol=0.01), samples, target, cv=cv_folds)    
    Logistic_mean[k-1]=accuracy_score(target,Logistic_predict)   
    Logistic_std[k-1]=Logistic_score.std()
    report=classification_report(target,Logistic_predict) #report
    lines = report.split('\n')
    NewLines=lines[5].split(' ')    
    Logistic_precision[k-1]=float(NewLines[9])   
    Logistic_recall[k-1]=float(NewLines[15])    
    Logistic_f1score[k-1]=float(NewLines[21])
    print("Logistic accuracy mean is",Logistic_mean[k-1])
    print("the Logistic standard deviation is",Logistic_std[k-1])
    print("the Logistic precision is",Logistic_precision[k-1])
    print("the Logistic recall is",Logistic_recall[k-1])
    print("the Logistic f1score is",Logistic_f1score[k-1],'\n')
    #print("the number is",i)
    #print("the getReport is",NewLines,'\n')
    #print("the report is",report,'\n')
    k=k+1
df=pd.DataFrame([[Logistic_mean[0],Logistic_mean[1],Logistic_mean[2],Logistic_mean[3],Logistic_mean[4],Logistic_mean[5],Logistic_mean[6],Logistic_mean[7],Logistic_mean[8],Logistic_mean[9],Logistic_mean[10],Logistic_mean[11]],[Logistic_precision[0],Logistic_precision[1],Logistic_precision[2],Logistic_precision[3],Logistic_precision[4],Logistic_precision[5],Logistic_precision[6],Logistic_precision[7],Logistic_precision[8],Logistic_precision[9],Logistic_precision[10],Logistic_precision[11]],[Logistic_recall[0],Logistic_recall[1],Logistic_recall[2],Logistic_recall[3],Logistic_recall[4],Logistic_recall[5],Logistic_recall[6],Logistic_recall[7],Logistic_recall[8],Logistic_recall[9],Logistic_recall[10],Logistic_recall[11]],[Logistic_f1score[0],Logistic_f1score[1],Logistic_f1score[2],Logistic_f1score[3],Logistic_f1score[4],Logistic_f1score[5],Logistic_f1score[6],Logistic_f1score[7],Logistic_f1score[8],Logistic_f1score[9],Logistic_f1score[10],Logistic_f1score[11]]],index=['accuracy','precision','recall','f1score'],columns=['Developer','Architect','ElementarySchoolLibrarian','Doctorsandhealthcareprofessionals','APA_Planning','AOAConnect','Futurist','ScienceStars','Top100CIOs','Tennis','TennisMale','Chemists'])
#print(Logistic_mean[0])
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
df.Developer.plot(kind='bar', color='firebrick', ax=ax, width=0.05, position=0)
df.Architect.plot(kind='bar', color='rosybrown', ax=ax, width=0.05, position=-1)
df.ElementarySchoolLibrarian.plot(kind='bar', color='blue', ax=ax, width=0.05, position=-2)
df.Doctorsandhealthcareprofessionals.plot(kind='bar', color='navy', ax=ax, width=0.05, position=-3)
df.APA_Planning.plot(kind='bar', color='green', ax=ax, width=0.05, position=-4)
df.AOAConnect.plot(kind='bar', color='darkgreen', ax=ax, width=0.05, position=-5)
df.Futurist.plot(kind='bar', color='yellow', ax=ax, width=0.05, position=-6)
df.ScienceStars.plot(kind='bar', color='y', ax=ax, width=0.05, position=-7)
df.Top100CIOs.plot(kind='bar', color='orange', ax=ax, width=0.05, position=-8)
df.Tennis.plot(kind='bar', color='tan', ax=ax, width=0.05, position=-9)
df.TennisMale.plot(kind='bar', color='gray', ax=ax, width=0.05, position=-10)
df.Chemists.plot(kind='bar', color='silver', ax=ax, width=0.05, position=-11)
ax.set_ylabel('percentage')
#ax.legend(())
plt.legend(['Developer','Architect','ElementarySchoolLibrarian','Doctorsandhealthcareprofessionals','@APA_Planning','@AOAConnect','Futurist','ScienceStars','Top100CIOs','Tennis','Tennis(Male)','Chemists'],loc='upper right',fontsize=8)
plt.title('Multibars')
plt.show()
