#-logistic,Author:ssb--
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('dataSample1.csv')
samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
from sklearn.linear_model import LogisticRegression
Logistic_predict = cross_validation.cross_val_predict(LogisticRegression(), samples, target, cv=cv_folds)

Logistic_score =cross_validation.cross_val_score(LogisticRegression(), samples, target, cv=cv_folds)
Logistic_mean=Logistic_score.mean()
Logistic_std=Logistic_score.std()
Logistic_precision=precision_score(target,Logistic_predict,average='weighted')
Logistic_recall=recall_score(target,Logistic_predict,average='weighted')
Logistic_f1score=f1_score(target,Logistic_predict,average='weighted')
print("Logistic accuracy mean is",Logistic_mean)
print("the Logistic standard deviation is",Logistic_std)
print("the Logistic precision is",Logistic_precision)
print("the Logistic recall is",Logistic_recall)
print("the Logistic f1score is",Logistic_f1score,'\n')

#randomforest
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
randomForest_predict = cross_validation.cross_val_predict(RandomForestClassifier(), samples, target, cv=cv_folds)
randomforest_score = cross_validation.cross_val_score(RandomForestClassifier(), samples, target, cv=cv_folds)
randomforest_mean=randomforest_score.mean()
randomforest_std=randomforest_score.std()
randomforest_precision=precision_score(target,randomForest_predict,average='weighted')
randomforest_recall=recall_score(target,randomForest_predict,average='weighted')
randomforest_f1score=f1_score(target,randomForest_predict,average='weighted')
print("randomforest accuracy mean is",randomforest_mean)
print("the randomforest standard deviation is",randomforest_std)
print("the randomforest precision is",randomforest_precision)
print("the randomforest recall is",randomforest_recall)
print("the randomforest f1score is",randomforest_f1score,'\n')

#AdaBoostClassifier
AdaBoost_predict = cross_validation.cross_val_predict(AdaBoostClassifier(), samples, target, cv=cv_folds)
AdaBoost_score = cross_validation.cross_val_score(AdaBoostClassifier(), samples, target, cv=cv_folds)
AdaBoost_mean=randomforest_score.mean()
AdaBoost_std=randomforest_score.std()
AdaBoost_precision=precision_score(target,AdaBoost_predict,average='weighted')
AdaBoost_recall=recall_score(target,AdaBoost_predict,average='weighted')
AdaBoost_f1score=f1_score(target,AdaBoost_predict,average='weighted')
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
bayes_precision=precision_score(target,bayes_predict,average='weighted')
bayes_recall=recall_score(target,bayes_predict,average='weighted')
bayes_f1score=f1_score(target,bayes_predict,average='weighted')
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
svm_precision=precision_score(target,svm_predict,average='weighted')
svm_recall=recall_score(target,svm_predict,average='weighted')
svm_f1score=f1_score(target,svm_predict,average='weighted')
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
