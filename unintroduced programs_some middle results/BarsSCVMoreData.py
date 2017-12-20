#-logistic,Author:ssb--
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv('moreDataSample.csv')
samples=df.loc[:,['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60','p61','p62','p63','p64','p65','p66','p67','p68','p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81','p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95','p97','p98','p99','p100','p101','p102','p103','p104','p105','p106','p107','p108','p109','p110','p111','p112','p113','p114','p115','p116','p117','p118','p119','p120','p121','p122','p123','p124','p125','p126','p127','p128','p129','p130','p131','p132','p133','p134']]
target=df.loc[:,'Profession']
cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
from sklearn.linear_model import LogisticRegression
Logistic_predict = cross_validation.cross_val_predict(LogisticRegression(), samples, target, cv=cv_folds)

Logistic_score =cross_validation.cross_val_score(LogisticRegression(), samples, target, cv=cv_folds)
report=classification_report(target,Logistic_predict) #report
Logistic_mean=accuracy_score(target,Logistic_predict)
#print(report)
Logistic_mean=Logistic_score.mean()
Logistic_std=Logistic_score.std()
lines = report.split('\n')
NewLines=lines[15].split(' ')
#print(NewLines)
Logistic_precision=float(NewLines[35])
Logistic_recall=float(NewLines[41])
Logistic_f1score=float(NewLines[47])
print("Logistic accuracy mean is",Logistic_mean)
print("the Logistic standard deviation is",Logistic_std)
print("the Logistic precision is",Logistic_precision)
print("the Logistic recall is",Logistic_recall)
#print("the report is",report,'\n')
print("the Logistic f1score is",Logistic_f1score,'\n')
#print("the getReport is",NewLines,'\n')

#randomforest
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
randomForest_predict = cross_validation.cross_val_predict(RandomForestClassifier(), samples, target, cv=cv_folds)
randomforest_score = cross_validation.cross_val_score(RandomForestClassifier(), samples, target, cv=cv_folds)
randomforest_mean=randomforest_score.mean()
randomforest_std=randomforest_score.std()
randomforest_report=classification_report(target,randomForest_predict) #report
randomforest_lines = report.split('\n')
randomforest_NewLines=randomforest_lines[15].split(' ')
randomforest_precision=float(randomforest_NewLines[35])
randomforest_recall=float(randomforest_NewLines[41])
randomforest_f1score=float(randomforest_NewLines[47])
print("randomforest accuracy mean is",randomforest_mean)
print("the randomforest standard deviation is",randomforest_std)
print("the randomforest precision is",randomforest_precision)
print("the randomforest recall is",randomforest_recall)
print("the randomforest f1score is",randomforest_f1score,'\n')

#AdaBoostClassifier
AdaBoost_predict = cross_validation.cross_val_predict(AdaBoostClassifier(), samples, target, cv=cv_folds)
AdaBoost_score = cross_validation.cross_val_score(AdaBoostClassifier(), samples, target, cv=cv_folds)
AdaBoost_mean=AdaBoost_score.mean()
AdaBoost_std=randomforest_score.std()
AdaBoost_report=classification_report(target,AdaBoost_predict) #report
AdaBoost_lines = report.split('\n')
AdaBoost_NewLines=AdaBoost_lines[15].split(' ')
AdaBoost_precision=float(AdaBoost_NewLines[35])
AdaBoost_recall=float(AdaBoost_NewLines[41])
AdaBoost_f1score=float(AdaBoost_NewLines[47])
print("AdaBoost accuracy mean is",AdaBoost_mean)
print("the AdaBoost standard deviation is",AdaBoost_std)
print("the AdaBoost precision is",AdaBoost_precision)
print("the AdaBoost recall is",AdaBoost_recall)
print("the AdaBoost f1score is",AdaBoost_f1score,'\n')

#NaiveBays
from sklearn.naive_bayes import GaussianNB
bayes_predict = cross_validation.cross_val_predict(GaussianNB(), samples, target, cv=cv_folds)
bayes_score = cross_validation.cross_val_score(GaussianNB(), samples, target, cv=cv_folds)
bayes_mean=bayes_score.mean()
bayes_std=bayes_score.std()
bayes_report=classification_report(target,bayes_predict) #report
bayes_lines = report.split('\n')
bayes_NewLines=bayes_lines[15].split(' ')
bayes_precision=float(bayes_NewLines[35])
bayes_recall=float(bayes_NewLines[41])
bayes_f1score=float(bayes_NewLines[47])
print("bayes accuracy mean is",bayes_mean)
print("the bayes standard deviation is",bayes_std)
print("the bayes precision is",bayes_precision)
print("the bayes recall is",bayes_recall)
print("the bayes f1score is",bayes_f1score,'\n')
#SVM
from sklearn.svm import SVC
svm_predict = cross_validation.cross_val_predict(SVC(), samples, target, cv=cv_folds)
svm_score = cross_validation.cross_val_score(SVC(), samples, target, cv=cv_folds)
svm_mean=svm_score.mean()
svm_std=svm_score.std()
svm_report=classification_report(target,svm_predict) #report
svm_lines = report.split('\n')
svm_NewLines=svm_lines[15].split(' ')
svm_precision=float(svm_NewLines[35])
svm_recall=float(svm_NewLines[41])
svm_f1score=float(svm_NewLines[47])
print("svm accuracy mean is",svm_mean)
print("the svm standard deviation is",svm_std)
print("the svm precision is",svm_precision)
print("the svm recall is",svm_recall)
print("the svm f1score is",svm_f1score,'\n')
#multibars
df=pd.DataFrame([[Logistic_mean,Logistic_precision,Logistic_recall,Logistic_f1score],[randomforest_mean,randomforest_precision,randomforest_recall,randomforest_f1score],[AdaBoost_mean,AdaBoost_precision,AdaBoost_recall,AdaBoost_f1score],[bayes_mean,bayes_precision,bayes_recall,bayes_f1score],[svm_mean,svm_precision,svm_recall,svm_f1score]],index=['Logistic','Randomforest','AdaBoost','bayes','SVM'],columns=['accuracy','precision','recall','f1score'])
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
df.accuracy.plot(kind='bar', color='red', ax=ax, width=0.2, position=0)
df.precision.plot(kind='bar', color='blue', ax=ax, width=0.2, position=-1)
df.recall.plot(kind='bar', color='green', ax=ax, width=0.2, position=-2)
df.f1score.plot(kind='bar', color='yellow', ax=ax, width=0.2, position=-3)
ax.set_ylabel('accuracy')
#ax.legend(())
plt.legend(['Logistic','randomforest','NaiveBays','SVM'],loc='upper right',fontsize=8)
plt.title('Multibars')
plt.show()
#errorbars
means   = [Logistic_mean,randomforest_mean,AdaBoost_mean,bayes_mean,svm_mean]         # Mean Data 
stds    = [Logistic_std,randomforest_std,AdaBoost_std,bayes_std,svm_std]            # Standard deviation Data
peakval = [Logistic_mean,randomforest_mean,AdaBoost_mean,bayes_mean,svm_mean] # String array of means
ind = np.arange(len(means))
width = 0.2
colours = ['green','green','green','green','green']
plt.figure()
plt.title('Errorbars')
for i in range(len(means)):
    plt.bar(ind[i],means[i],width,color=colours[i],align='center',yerr=stds[i],ecolor='k')
plt.ylabel('Accuracy mean(%)')
plt.xticks(ind,('Logistic','Randomforest','AdaBoost','Bayes','SVM'))
def autolabel(bars,peakval):
#def autolabel(bars):
    for ii,bar in enumerate(bars):
        height = bars[ii]
        plt.text(ind[ii], height-5, '%s'% (peakval[ii]), ha='center', va='bottom')
autolabel(means,peakval)
#autolabel(means)
plt.show()
