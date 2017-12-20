#-compare with different LRs upon moredata,Author:ssb--
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
#simple LR
df=pd.read_csv('moreDataSample.csv')
samples=df.loc[:,['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60','p61','p62','p63','p64','p65','p66','p67','p68','p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81','p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95','p97','p98','p99','p100','p101','p102','p103','p104','p105','p106','p107','p108','p109','p110','p111','p112','p113','p114','p115','p116','p117','p118','p119','p120','p121','p122','p123','p124','p125','p126','p127','p128','p129','p130','p131','p132','p133','p134']]
target=df.loc[:,'Profession']
cv_folds = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=False, random_state=0)
from sklearn.linear_model import LogisticRegression
Logistic_predict = cross_validation.cross_val_predict(LogisticRegression(), samples, target, cv=cv_folds)
Logistic_score =cross_validation.cross_val_score(LogisticRegression(), samples, target, cv=cv_folds)
Logistic_accuracy=accuracy_score(target,Logistic_predict)
Logistic_std=Logistic_score.std()
report=classification_report(target,Logistic_predict) #report
lines = report.split('\n')
NewLines=lines[15].split(' ')
Logistic_precision=float(NewLines[9])
Logistic_recall=float(NewLines[15])
Logistic_f1score=float(NewLines[21])
print("Logistic accuracy accuracy is",Logistic_accuracy)
print("the Logistic standard deviation is",Logistic_std)
print("the Logistic precision is",Logistic_precision)
print("the Logistic recall is",Logistic_recall)
print("the Logistic f1score is",Logistic_f1score,'\n')
#Simple SR L1
Logistic_predict_L1 = cross_validation.cross_val_predict(LogisticRegression(penalty='l1',tol=0.01), samples, target, cv=cv_folds)
Logistic_score_L1 =cross_validation.cross_val_score(LogisticRegression(penalty='l1',tol=0.01), samples, target, cv=cv_folds)
Logistic_accuracy_L1=accuracy_score(target,Logistic_predict_L1)
Logistic_std_L1=Logistic_score_L1.std()
report_L1=classification_report(target,Logistic_predict_L1) #report
lines_L1 = report_L1.split('\n')
NewLines_L1=lines_L1[15].split(' ')
Logistic_precision_L1=float(NewLines_L1[9])
Logistic_recall_L1=float(NewLines_L1[15])
Logistic_f1score_L1=float(NewLines_L1[21])
print("Logistic accuracy accuracy is",Logistic_accuracy_L1)
print("the Logistic standard deviation is",Logistic_std_L1)
print("the Logistic precision is",Logistic_precision_L1)
print("the Logistic recall is",Logistic_recall_L1)
print("the Logistic f1score is",Logistic_f1score_L1,'\n')

#Simple SR L2
Logistic_predict_L2 = cross_validation.cross_val_predict(LogisticRegression(penalty='l2',tol=0.01), samples, target, cv=cv_folds)
Logistic_score_L2 =cross_validation.cross_val_score(LogisticRegression(penalty='l2',tol=0.01), samples, target, cv=cv_folds)
Logistic_accuracy_L2=accuracy_score(target,Logistic_predict_L2)
Logistic_std_L2=Logistic_score_L2.std()
report_L2=classification_report(target,Logistic_predict_L2) #report
lines_L2 = report_L2.split('\n')
NewLines_L2=lines_L2[15].split(' ')
Logistic_precision_L2=float(NewLines_L2[9])
Logistic_recall_L2=float(NewLines_L2[15])
Logistic_f1score_L2=float(NewLines_L2[21])
print("Logistic accuracy accuracy is",Logistic_accuracy_L2)
print("the Logistic standard deviation is",Logistic_std_L2)
print("the Logistic precision is",Logistic_precision_L2)
print("the Logistic recall is",Logistic_recall_L2)
print("the Logistic f1score is",Logistic_f1score_L2,'\n')
#LR one vs rest
k=1
Logistic_accuracy_onevsrest =['','','','','','','','','','','','']
Logistic_std_onevsrest =['','','','','','','','','','','','']
Logistic_precision_onevsrest =['','','','','','','','','','','','']
Logistic_recall_onevsrest =['','','','','','','','','','','','']
Logistic_f1score_onevsrest =['','','','','','','','','','','','']
for i in range(1,13,1):
    fname="{}.csv".format(k)
    df_onevsrest=pd.read_csv(fname)
    samples_onevsrest=df_onevsrest.loc[:,['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60','p61','p62','p63','p64','p65','p66','p67','p68','p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81','p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95','p97','p98','p99','p100','p101','p102','p103','p104','p105','p106','p107','p108','p109','p110','p111','p112','p113','p114','p115','p116','p117','p118','p119','p120','p121','p122','p123','p124','p125','p126','p127','p128','p129','p130','p131','p132','p133','p134']]
    target_onevsrest=df_onevsrest.loc[:,'Profession']
    cv_folds_onevsrest = cross_validation.StratifiedKFold(target_onevsrest, n_folds=5, shuffle=False, random_state=0)
    #logistic
    from sklearn.linear_model import LogisticRegression
    Logistic_predict_onevsrest = cross_validation.cross_val_predict(LogisticRegression(), samples_onevsrest, target_onevsrest, cv=cv_folds_onevsrest)
    Logistic_score_onevsrest =cross_validation.cross_val_score(LogisticRegression(), samples_onevsrest, target_onevsrest, cv=cv_folds_onevsrest)    
    Logistic_accuracy_onevsrest[k-1]=accuracy_score(target_onevsrest,Logistic_predict_onevsrest)   
    Logistic_std_onevsrest[k-1]=Logistic_score_onevsrest.std()
    report_onevsrest=classification_report(target_onevsrest,Logistic_predict_onevsrest) #report
    lines_onevsrest = report_onevsrest.split('\n')
    NewLines_onevsrest=lines_onevsrest[5].split(' ')    
    Logistic_precision_onevsrest[k-1]=float(NewLines_onevsrest[9])   
    Logistic_recall_onevsrest[k-1]=float(NewLines_onevsrest[15])    
    Logistic_f1score_onevsrest[k-1]=float(NewLines_onevsrest[21])
    k=k+1
count_accuracy=0
count_std=0
count_precision=0
count_recall=0
count_f1score=0
for k in range(1,13,1):
    Logistic_accuracy_onevsrest_sum=count_accuracy
    Logistic_std_onevsrest_sum=count_std
    Logistic_precision_onevsrest_sum=count_precision
    Logistic_recall_onevsrest_sum=count_recall
    Logistic_f1score_onevsrest_sum=count_f1score
    
    Logistic_accuracy_onevsrest_sum=Logistic_accuracy_onevsrest[k-1]+Logistic_accuracy_onevsrest_sum
    Logistic_std_onevsrest_sum=Logistic_std_onevsrest[k-1]+Logistic_std_onevsrest_sum
    Logistic_precision_onevsrest_sum=Logistic_precision_onevsrest[k-1]+Logistic_precision_onevsrest_sum
    Logistic_recall_onevsrest_sum=Logistic_recall_onevsrest[k-1]+Logistic_recall_onevsrest_sum
    Logistic_f1score_onevsrest_sum=Logistic_f1score_onevsrest[k-1]+Logistic_f1score_onevsrest_sum
    
    count_accuracy=Logistic_accuracy_onevsrest_sum
    count_std=Logistic_std_onevsrest_sum
    count_precision=Logistic_precision_onevsrest_sum
    count_recall=Logistic_recall_onevsrest_sum
    count_f1score=Logistic_f1score_onevsrest_sum
    
accuracy_avg=Logistic_accuracy_onevsrest_sum/12
std_avg=Logistic_std_onevsrest_sum/12
precision_avg=Logistic_precision_onevsrest_sum/12
recall_avg=Logistic_recall_onevsrest_sum/12
f1score_avg=Logistic_f1score_onevsrest_sum/12
print("the average accuracy is",accuracy_avg)
print("the average std is",std_avg)
print("the average precision is",precision_avg)
print("the average recall is",recall_avg)
print("the average f1score is",f1score_avg)
#LR w.L1
k_L1=1
Logistic_accuracy_onevsrest_L1 =['','','','','','','','','','','','']
Logistic_std_onevsrest_L1 =['','','','','','','','','','','','']
Logistic_precision_onevsrest_L1 =['','','','','','','','','','','','']
Logistic_recall_onevsrest_L1 =['','','','','','','','','','','','']
Logistic_f1score_onevsrest_L1 =['','','','','','','','','','','','']
for j in range(1,13,1):
    fname_L1="{}.csv".format(k_L1)
    df_onevsrest_L1=pd.read_csv(fname_L1)
    samples_onevsrest_L1=df_onevsrest_L1.loc[:,['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60','p61','p62','p63','p64','p65','p66','p67','p68','p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81','p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95','p97','p98','p99','p100','p101','p102','p103','p104','p105','p106','p107','p108','p109','p110','p111','p112','p113','p114','p115','p116','p117','p118','p119','p120','p121','p122','p123','p124','p125','p126','p127','p128','p129','p130','p131','p132','p133','p134']]
    target_onevsrest_L1=df_onevsrest_L1.loc[:,'Profession']
    cv_folds_onevsrest_L1 = cross_validation.StratifiedKFold(target_onevsrest_L1, n_folds=5, shuffle=False, random_state=0)
    #logistic
    from sklearn.linear_model import LogisticRegression
    Logistic_predict_onevsrest_L1 = cross_validation.cross_val_predict(LogisticRegression(penalty='l1',tol=0.01), samples_onevsrest_L1, target_onevsrest_L1, cv=cv_folds_onevsrest_L1)
    Logistic_score_onevsrest_L1 =cross_validation.cross_val_score(LogisticRegression(penalty='l1',tol=0.01), samples_onevsrest_L1, target_onevsrest_L1, cv=cv_folds_onevsrest_L1)    
    Logistic_accuracy_onevsrest_L1[k_L1-1]=accuracy_score(target_onevsrest_L1,Logistic_predict_onevsrest_L1)   
    Logistic_std_onevsrest_L1[k_L1-1]=Logistic_score_onevsrest_L1.std()
    report_onevsrest_L1=classification_report(target_onevsrest_L1,Logistic_predict_onevsrest_L1) #report
    lines_onevsrest_L1 = report_onevsrest_L1.split('\n')
    NewLines_onevsrest_L1=lines_onevsrest_L1[5].split(' ')    
    Logistic_precision_onevsrest_L1[k_L1-1]=float(NewLines_onevsrest_L1[9])   
    Logistic_recall_onevsrest_L1[k_L1-1]=float(NewLines_onevsrest_L1[15])    
    Logistic_f1score_onevsrest_L1[k_L1-1]=float(NewLines_onevsrest_L1[21])
    k_L1=k_L1+1
count_accuracy_L1=0
count_std_L1=0
count_precision_L1=0
count_recall_L1=0
count_f1score_L1=0
for k_L1 in range(1,13,1):
    Logistic_accuracy_onevsrest_sum_L1=count_accuracy_L1
    Logistic_std_onevsrest_sum_L1=count_std_L1
    Logistic_precision_onevsrest_sum_L1=count_precision_L1
    Logistic_recall_onevsrest_sum_L1=count_recall_L1
    Logistic_f1score_onevsrest_sum_L1=count_f1score_L1
    
    Logistic_accuracy_onevsrest_sum_L1=Logistic_accuracy_onevsrest_L1[k-1]+Logistic_accuracy_onevsrest_sum_L1
    Logistic_std_onevsrest_sum_L1=Logistic_std_onevsrest_L1[k-1]+Logistic_std_onevsrest_sum_L1
    Logistic_precision_onevsrest_sum_L1=Logistic_precision_onevsrest_L1[k-1]+Logistic_precision_onevsrest_sum_L1
    Logistic_recall_onevsrest_sum_L1=Logistic_recall_onevsrest_L1[k-1]+Logistic_recall_onevsrest_sum_L1
    Logistic_f1score_onevsrest_sum_L1=Logistic_f1score_onevsrest_L1[k-1]+Logistic_f1score_onevsrest_sum_L1
    
    count_accuracy_L1=Logistic_accuracy_onevsrest_sum_L1
    count_std_L1=Logistic_std_onevsrest_sum_L1
    count_precision_L1=Logistic_precision_onevsrest_sum_L1
    count_recall_L1=Logistic_recall_onevsrest_sum_L1
    count_f1score_L1=Logistic_f1score_onevsrest_sum_L1
    
accuracy_avg_L1=Logistic_accuracy_onevsrest_sum_L1/12
std_avg_L1=Logistic_std_onevsrest_sum_L1/12
precision_avg_L1=Logistic_precision_onevsrest_sum_L1/12
recall_avg_L1=Logistic_recall_onevsrest_sum_L1/12
f1score_avg_L1=Logistic_f1score_onevsrest_sum_L1/12
print("the average accuracy is",accuracy_avg_L1)
print("the average std is",std_avg_L1)
print("the average precision is",precision_avg_L1)
print("the average recall is",recall_avg_L1)
print("the average f1score is",f1score_avg_L1)

#LR w.L2
k_L2=1
Logistic_accuracy_onevsrest_L2 =['','','','','','','','','','','','']
Logistic_std_onevsrest_L2 =['','','','','','','','','','','','']
Logistic_precision_onevsrest_L2 =['','','','','','','','','','','','']
Logistic_recall_onevsrest_L2 =['','','','','','','','','','','','']
Logistic_f1score_onevsrest_L2 =['','','','','','','','','','','','']
for j in range(1,13,1):
    fname_L2="{}.csv".format(k_L2)
    df_onevsrest_L2=pd.read_csv(fname_L2)
    samples_onevsrest_L2=df_onevsrest_L2.loc[:,['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60','p61','p62','p63','p64','p65','p66','p67','p68','p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81','p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95','p97','p98','p99','p100','p101','p102','p103','p104','p105','p106','p107','p108','p109','p110','p111','p112','p113','p114','p115','p116','p117','p118','p119','p120','p121','p122','p123','p124','p125','p126','p127','p128','p129','p130','p131','p132','p133','p134']]
    target_onevsrest_L2=df_onevsrest_L2.loc[:,'Profession']
    cv_folds_onevsrest_L2 = cross_validation.StratifiedKFold(target_onevsrest_L2, n_folds=5, shuffle=False, random_state=0)
    #logistic
    from sklearn.linear_model import LogisticRegression
    Logistic_predict_onevsrest_L2 = cross_validation.cross_val_predict(LogisticRegression(penalty='l2',tol=0.01), samples_onevsrest_L2, target_onevsrest_L2, cv=cv_folds_onevsrest_L2)
    Logistic_score_onevsrest_L2 =cross_validation.cross_val_score(LogisticRegression(penalty='l2',tol=0.01), samples_onevsrest_L2, target_onevsrest_L2, cv=cv_folds_onevsrest_L2)    
    Logistic_accuracy_onevsrest_L2[k_L2-1]=accuracy_score(target_onevsrest_L2,Logistic_predict_onevsrest_L2)   
    Logistic_std_onevsrest_L2[k_L2-1]=Logistic_score_onevsrest_L2.std()
    report_onevsrest_L2=classification_report(target_onevsrest_L2,Logistic_predict_onevsrest_L2) #report
    lines_onevsrest_L2 = report_onevsrest_L2.split('\n')
    NewLines_onevsrest_L2=lines_onevsrest_L2[5].split(' ')    
    Logistic_precision_onevsrest_L2[k_L2-1]=float(NewLines_onevsrest_L2[9])   
    Logistic_recall_onevsrest_L2[k_L2-1]=float(NewLines_onevsrest_L2[15])    
    Logistic_f1score_onevsrest_L2[k_L2-1]=float(NewLines_onevsrest_L2[21])
    k_L2=k_L2+1
count_accuracy_L2=0
count_std_L2=0
count_precision_L2=0
count_recall_L2=0
count_f1score_L2=0
for k_L2 in range(1,13,1):
    Logistic_accuracy_onevsrest_sum_L2=count_accuracy_L2
    Logistic_std_onevsrest_sum_L2=count_std_L2
    Logistic_precision_onevsrest_sum_L2=count_precision_L2
    Logistic_recall_onevsrest_sum_L2=count_recall_L2
    Logistic_f1score_onevsrest_sum_L2=count_f1score_L2
    
    Logistic_accuracy_onevsrest_sum_L2=Logistic_accuracy_onevsrest_L2[k-1]+Logistic_accuracy_onevsrest_sum_L2
    Logistic_std_onevsrest_sum_L2=Logistic_std_onevsrest_L2[k-1]+Logistic_std_onevsrest_sum_L2
    Logistic_precision_onevsrest_sum_L2=Logistic_precision_onevsrest_L2[k-1]+Logistic_precision_onevsrest_sum_L2
    Logistic_recall_onevsrest_sum_L2=Logistic_recall_onevsrest_L2[k-1]+Logistic_recall_onevsrest_sum_L2
    Logistic_f1score_onevsrest_sum_L2=Logistic_f1score_onevsrest_L2[k-1]+Logistic_f1score_onevsrest_sum_L2
    
    count_accuracy_L2=Logistic_accuracy_onevsrest_sum_L2
    count_std_L2=Logistic_std_onevsrest_sum_L2
    count_precision_L2=Logistic_precision_onevsrest_sum_L2
    count_recall_L2=Logistic_recall_onevsrest_sum_L2
    count_f1score_L2=Logistic_f1score_onevsrest_sum_L2
    
accuracy_avg_L2=Logistic_accuracy_onevsrest_sum_L2/12
std_avg_L2=Logistic_std_onevsrest_sum_L2/12
precision_avg_L2=Logistic_precision_onevsrest_sum_L2/12
recall_avg_L2=Logistic_recall_onevsrest_sum_L2/12
f1score_avg_L2=Logistic_f1score_onevsrest_sum_L2/12
print("the average accuracy is",accuracy_avg_L2)
print("the average std is",std_avg_L2)
print("the average precision is",precision_avg_L2)
print("the average recall is",recall_avg_L2)
print("the average f1score is",f1score_avg_L2)
#plot
df=pd.DataFrame([[Logistic_accuracy,Logistic_accuracy_L1,Logistic_accuracy_L2,accuracy_avg,accuracy_avg_L1,accuracy_avg_L2],[Logistic_precision,Logistic_precision_L1,Logistic_precision_L2,precision_avg,precision_avg_L1,precision_avg_L2],[Logistic_recall,Logistic_recall_L1,Logistic_recall_L2,recall_avg,recall_avg_L1,recall_avg_L2],[Logistic_f1score,Logistic_f1score_L1,Logistic_f1score_L2,f1score_avg,f1score_avg_L1,f1score_avg_L2]],index=['accuracy','precision','recall','f1score'],columns=['SimpleLR','LR_L1','LR_L2','LR_Onevsrest','LR_Onevsrest_L1','LR_Onevsrest_L2'])
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
df.SimpleLR.plot(kind='bar', color='firebrick', ax=ax, width=0.05, position=0)
df.LR_L1.plot(kind='bar', color='rosybrown', ax=ax, width=0.05, position=-1)
df.LR_L2.plot(kind='bar', color='blue', ax=ax, width=0.05, position=-2)
df.LR_Onevsrest.plot(kind='bar', color='navy', ax=ax, width=0.05, position=-3)
df.LR_Onevsrest_L1.plot(kind='bar', color='green', ax=ax, width=0.05, position=-4)
df.LR_Onevsrest_L2.plot(kind='bar', color='darkgreen', ax=ax, width=0.05, position=-5)
ax.set_ylabel('percentage')
plt.legend(['SimpleLR','LR_L1','LR_L2','LR_Onevsrest','LR_Onevsrest_L1','LR_Onevsrest_L2'],loc='upper right',fontsize=8)
plt.title('Multibars')
plt.show()
