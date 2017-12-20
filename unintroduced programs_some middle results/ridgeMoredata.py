from sklearn import linear_model
from sklearn.linear_model import Ridge,Lasso
from sklearn import metrics, cross_validation
from sklearn import datasets
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import pandas as pd
import numpy as np

df=pd.read_csv('moreDataSample.csv')
samples=df.loc[:,['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60','p61','p62','p63','p64','p65','p66','p67','p68','p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81','p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95','p97','p98','p99','p100','p101','p102','p103','p104','p105','p106','p107','p108','p109','p110','p111','p112','p113','p114','p115','p116','p117','p118','p119','p120','p121','p122','p123','p124','p125','p126','p127','p128','p129','p130','p131','p132','p133','p134']]
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



