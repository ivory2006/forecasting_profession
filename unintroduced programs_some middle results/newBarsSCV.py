#-multi-algorithm,Author:ssb--
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
import pprint
import matplotlib.pyplot as plt
df=pd.read_csv('dataSample1.csv')
samples=df.loc[:,['Openness','Conscientousness','Extraversion','Agreeableness','Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence']]
target=df.loc[:,'Profession']
cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
from sklearn.linear_model import LogisticRegression
Logistic_predict = cross_validation.cross_val_predict(LogisticRegression(solver='newton-cg',multi_class='multinomial'), samples, target, cv=cv_folds)
Logistic_score =cross_validation.cross_val_score(LogisticRegression(solver='newton-cg',multi_class='multinomial'), samples, target, cv=cv_folds)
#Logistic_predict = cross_validation.cross_val_predict(LogisticRegression(), samples, target, cv=cv_folds)
#Logistic_score =cross_validation.cross_val_score(LogisticRegression(), samples, target, cv=cv_folds)
Logistic_mean=accuracy_score(target,Logistic_predict)
Logistic_std=Logistic_score.std()
report=classification_report(target,Logistic_predict) #report
lines = report.split('\n')
#pprint.pprint(lines)
#pprint.pprint(lines[15])
NewLines=lines[15].split(' ')  #dataSample1:15
Logistic_precision=float(NewLines[9])
Logistic_recall=float(NewLines[15])
Logistic_f1score=float(NewLines[21])
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
randomforest_report=classification_report(target,randomForest_predict) #report
randomforest_lines = randomforest_report.split('\n')
randomforest_NewLines=randomforest_lines[15].split(' ')
randomforest_precision=float(randomforest_NewLines[9])
randomforest_recall=float(randomforest_NewLines[15])
randomforest_f1score=float(randomforest_NewLines[21])
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
AdaBoost_report=classification_report(target,AdaBoost_predict) #report
AdaBoost_lines = AdaBoost_report.split('\n')
AdaBoost_NewLines=AdaBoost_lines[15].split(' ')
AdaBoost_precision=float(AdaBoost_NewLines[9])
AdaBoost_recall=float(AdaBoost_NewLines[15])
AdaBoost_f1score=float(AdaBoost_NewLines[21])
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
bayes_lines = bayes_report.split('\n')
bayes_NewLines=bayes_lines[15].split(' ')
bayes_precision=float(bayes_NewLines[9])
bayes_recall=float(bayes_NewLines[15])
bayes_f1score=float(bayes_NewLines[21])
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
svm_lines = svm_report.split('\n')
svm_NewLines=svm_lines[15].split(' ')
svm_precision=float(svm_NewLines[9])
svm_recall=float(svm_NewLines[15])
svm_f1score=float(svm_NewLines[21])
print("svm accuracy mean is",svm_mean)
print("the svm standard deviation is",svm_std)
print("the svm precision is",svm_precision)
print("the svm recall is",svm_recall)
print("the svm f1score is",svm_f1score,'\n')
#cart
from sklearn.tree import DecisionTreeClassifier
tree_predict = cross_validation.cross_val_predict(DecisionTreeClassifier(), samples, target, cv=cv_folds)
tree_score = cross_validation.cross_val_score(DecisionTreeClassifier(), samples, target, cv=cv_folds)
tree_mean=tree_score.mean()
tree_std=tree_score.std()
tree_report=classification_report(target,tree_predict) #report
tree_lines = tree_report.split('\n')
tree_NewLines=tree_lines[15].split(' ')
tree_precision=float(tree_NewLines[9])
tree_recall=float(tree_NewLines[15])
tree_f1score=float(tree_NewLines[21])
print("tree accuracy mean is",tree_mean)
print("the tree standard deviation is",tree_std)
print("the tree precision is",tree_precision)
print("the tree recall is",tree_recall)
print("the tree f1score is",tree_f1score,'\n')

#multibars
df=pd.DataFrame([[Logistic_mean,Logistic_precision,Logistic_recall,Logistic_f1score],[randomforest_mean,randomforest_precision,randomforest_recall,randomforest_f1score],[AdaBoost_mean,AdaBoost_precision,AdaBoost_recall,AdaBoost_f1score],[bayes_mean,bayes_precision,bayes_recall,bayes_f1score],[svm_mean,svm_precision,svm_recall,svm_f1score]],index=['Logistic','Randomforest','AdaBoost','bayes','SVM'],columns=['accuracy','precision','recall','f1score'])
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
df.accuracy.plot(kind='bar', color='red', ax=ax, width=0.2, position=0)
df.precision.plot(kind='bar', color='blue', ax=ax, width=0.2, position=-1)
df.recall.plot(kind='bar', color='green', ax=ax, width=0.2, position=-2)
df.f1score.plot(kind='bar', color='yellow', ax=ax, width=0.2, position=-3)
ax.set_ylabel('percent')
plt.legend(['accuracy','presicion','recall','f1score'],loc='upper right',fontsize=8)
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

