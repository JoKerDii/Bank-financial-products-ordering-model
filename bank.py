# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:23:25 2018

@author: dizhen
"""

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

df=pd.read_csv('bank-full.csv',sep=';')
df.head(10)

#################################################### Data cleaning ####################################################

# s stores all value_counts
# value_counts() counts unique objects
s=[]
for i in df.columns:
    s.append(df[str(i)].value_counts())

# Transform string to number 

# s[1].index is an object and should be transformed to list
s_index1=list(s[1].index)
# Find the object that equals to one in s[].index in every column, then replace the value with the index of it in s list.
for i in s[1].index:
    df.loc[df[str(df.columns[1])]==str(i),str(df.columns[1])]=s_index1.index(str(i))
    
s_index2=list(s[2].index)
for i in s[2].index:
    df.loc[df[str(df.columns[2])]==str(i),str(df.columns[2])]=s_index2.index(str(i))
    
s_index3=list(s[3].index)
for i in s[3].index:
    df.loc[df[str(df.columns[3])]==str(i),str(df.columns[3])]=s_index3.index(str(i))
    
s_index4=list(s[4].index)
for i in s[4].index:
    df.loc[df[str(df.columns[4])]==str(i),str(df.columns[4])]=s_index4.index(str(i))
    
s_index6=list(s[6].index)
for i in s[6].index:
    df.loc[df[str(df.columns[6])]==str(i),str(df.columns[6])]=s_index6.index(str(i))
    
s_index7=list(s[7].index)
for i in s[7].index:
    df.loc[df[str(df.columns[7])]==str(i),str(df.columns[7])]=s_index7.index(str(i))
    
s_index8=list(s[8].index)
for i in s[8].index:
    df.loc[df[str(df.columns[8])]==str(i),str(df.columns[8])]=s_index8.index(str(i))
    
s_index10=list(s[10].index)
for i in s[10].index:
    df.loc[df[str(df.columns[10])]==str(i),str(df.columns[10])]=s_index10.index(str(i))
    
s_index15=list(s[15].index)
for i in s[15].index:
    df.loc[df[str(df.columns[15])]==str(i),str(df.columns[15])]=s_index15.index(str(i))
    
s_index16=list(s[16].index)
for i in s[16].index:
    df.loc[df[str(df.columns[16])]==str(i),str(df.columns[16])]=s_index16.index(str(i))


# drop NA, Keep rows with at least 17 Non-Null values
df=df.dropna(thresh=17)
# fill NA
df=df.fillna(method='ffill',limit=3)
df.apply(lambda x:((x-x.mean())/x.var()))

# select X and y from dataframe
X=df.iloc[:,0:16]
# if y=df.iloc[:,16：17], we get a dataframe，otherwise we get a series. Here is a series object
y=df.iloc[:,16]
# 25% as training data for default， use 'test_size' argument to give a percentage to split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#################################################### logistic regression ####################################################

log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
pred_log=log_reg.predict(X_test)
# Use 'predict_proba' to get AUC, return the probability in every classification (if it is dichotomy, there are two columns)
pred_proba_log=log_reg.predict_proba(X_test)

print("Logistic classification results:")
# accuracy_score reflects the ratio of correct positive to predicted positive
print("accuracy_score:",accuracy_score(y_test,pred_log))
# precision_score reflects the prediction precision 
print("precision_score:",precision_score(y_test,pred_log))
# recall_score reflects the ratio of correct positive to true positive
print("recall_score",recall_score(y_test,pred_log))
print("auc:",roc_auc_score(y_test,pred_proba_log[:,1]))
print("f1_score(weighted):",f1_score(y_test,pred_log,average='weighted'))
print("f1_score(macro):",f1_score(y_test,pred_log,average='macro'))
print("f1_score(micro):",f1_score(y_test,pred_log,average='micro'))
print("f1_score(None):",f1_score(y_test,pred_log))

####################################################  svm ####################################################


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# method without pipeline 
# scaler=StandardScaler()
# scaler.fit(df)
# svm_clf=SVC(C=1,probability=True,verbose=1)

# standardize by column
svm_clf=Pipeline((
        ('scaler',StandardScaler()),
        ('linear_svc',LinearSVC(C=1,loss='hinge'))
        ))
svm_clf.fit(X_train,y_train)
pred_svm=svm_clf.predict(X_test)

print("svm classification result")
print("accuracy_score:",accuracy_score(y_test,pred_svm))
print("precision_score:",precision_score(y_test,pred_svm))
print("recall_score",recall_score(y_test,pred_svm))
#print("auc:",roc_auc_score(y_test,pred_proba_svm[:,1]))#auc
print("f1_score(weighted):",f1_score(y_test,pred_svm,average='weighted'))
print("f1_score(macro):",f1_score(y_test,pred_svm,average='macro'))
print("f1_score(micro):",f1_score(y_test,pred_svm,average='micro'))
print("f1_score(None):",f1_score(y_test,pred_svm))

#################################################### random forest ####################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("random forest classification result:")
rnd_clf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1)
rnd_clf.fit(X_train,y_train)
pred_rf=rnd_clf.predict(X_test)
pred_proba_rf=rnd_clf.predict_proba(X_test)
print("accuracy_score:",accuracy_score(y_test,pred_rf))
print("precision_score:",precision_score(y_test,pred_rf))
print("recall_score",recall_score(y_test,pred_rf))
print("auc:",roc_auc_score(y_test,pred_proba_rf[:,1]))#auc
print("f1_score(weighted):",f1_score(y_test,pred_rf,average='weighted'))
print("f1_score(macro):",f1_score(y_test,pred_rf,average='macro'))
print("f1_score(micro):",f1_score(y_test,pred_rf,average='micro'))
print("f1_score(None):",f1_score(y_test,pred_rf))

import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np


#################################################### stacking classifier ####################################################

print("Stacking:\n")
clf1=KNeighborsClassifier(n_neighbors=1)
clr2=RandomForestClassifier(random_state=1)
clf3=GaussianNB()
lr=LogisticRegression()#logistics
sclf=StackingClassifier(classifiers=[clf1,clr2,clf3],meta_classifier=lr)

print('3-fold cross validation:\n')

for clf,label in zip([clf1,clr2,clf3,sclf],
                     ['KNN',
                      'Random Forest',
                      'Naive Bayes',
                      'StackingClassifier']):
    scores_acc=model_selection.cross_val_score(clf,X,y,cv=3,scoring='accuracy')
    scores_auc=model_selection.cross_val_score(clf,X,y,cv=3,scoring='roc_auc')
    scores_f1=model_selection.cross_val_score(clf,X,y,cv=3,scoring='f1')
    scores_f1_macro=model_selection.cross_val_score(clf,X,y,cv=3,scoring='f1_macro')    
    scores_f1_micro=model_selection.cross_val_score(clf,X,y,cv=3,scoring='f1_micro')
    scores_f1_weighted=model_selection.cross_val_score(clf,X,y,cv=3,scoring='f1_weighted')
    print("Accuracy:%0.2f(+/- %0.2f) [%s]\nAuc:%0.2f(+/- %0.2f) [%s]\nf1:%0.2f(+/- %0.2f) [%s]\nf1_micro:%0.2f(+/- %0.2f) [%s]\nf1_macro:%0.2f(+/- %0.2f) [%s]\nf1_weighted:%0.2f(+/- %0.2f) [%s]\n"
          %(scores_acc.mean(),scores_acc.std(),label,
            scores_auc.mean(),scores_auc.std(),label,
            scores_f1.mean(),scores_f1.std(),label,
            scores_f1_micro.mean(),scores_f1_micro.std(),label,
            scores_f1_macro.mean(),scores_f1_macro.std(),label,
            scores_f1_weighted.mean(),scores_f1_weighted.std(),label
            ))
    
    """
print("normal stacking：\n")
import numpy as np

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
#initalizing classifiers
clf1=LogisticRegression(random_state=0)
clf2=RandomForestClassifier(random_state=0)
clf3=SVC(random_state=0,probability=True)
eclf=EnsembleVoteClassifier(clfs=[clf1,clf2,clf3],weights=[2,1,1],voting='soft')

#loading some example data
for clf,lab in zip([clf1,clf2,clf3,eclf],
                   ['Logistic Regression','Random Forest','Naive Bayes','Ensemble']):
    scores_acc=model_selection.cross_val_score(clf,X,y,cv=3,scoring='accuracy')
    scores_auc=model_selection.cross_val_score(clf,X,y,cv=3,scoring='roc_auc')
    scores_f1=model_selection.cross_val_score(clf,X,y,cv=3,scoring='f1')
    scores_f1_macro=model_selection.cross_val_score(clf,X,y,cv=3,scoring='f1_macro')    
    scores_f1_micro=model_selection.cross_val_score(clf,X,y,cv=3,scoring='f1_micro')
    scores_f1_weighted=model_selection.cross_val_score(clf,X,y,cv=3,scoring='f1_weighted')
    print("Accuracy:%0.2f(+/- %0.2f) [%s]\nAuc:%0.2f(+/- %0.2f) [%s]\nf1:%0.2f(+/- %0.2f) [%s]\nf1_micro:%0.2f(+/- %0.2f) [%s]\nf1_macro:%0.2f(+/- %0.2f) [%s]\nf1_weighted:%0.2f(+/- %0.2f) [%s]\n"
          %(scores_acc.mean(),scores_acc.std(),lab,
            scores_auc.mean(),scores_auc.std(),lab,
            scores_f1.mean(),scores_f1.std(),lab,
            scores_f1_micro.mean(),scores_f1_micro.std(),lab,
            scores_f1_macro.mean(),scores_f1_macro.std(),lab,
            scores_f1_weighted.mean(),scores_f1_weighted.std(),lab
            ))"""

#################################################### ROC Curve ####################################################


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve,auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt


# roc curve
print('ROC curve')
kfold=StratifiedKFold(n_splits=2,random_state=1)
'''pipeline,turtle or list is outermost，inside must be turtle'''
pipe_lr=Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=1)),('clf',LogisticRegression(random_state=1))])
for i, (train,test) in enumerate(kfold.split(X_train,y_train)): 
    prob=pipe_lr.fit(X_train.iloc[train],y_train.iloc[train]).predict_proba(X_train.iloc[test])
    fpr,tqr,thresholds=roc_curve(y_train.iloc[test],prob[:,1],pos_label=1)
    roc_auc=auc(fpr,tqr)
    plt.plot(fpr,tqr,label='ROC fold:{},auc:{}'.format(i,roc_auc))


