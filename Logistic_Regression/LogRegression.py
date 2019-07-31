#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:44:57 2019

@author: Yesser
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sb

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# import data
data0 = pd.read_csv('data.csv')
print(data0.info())
print(data0.isnull().sum())
data0.dropna(inplace=True)

print(data0.shape)
print(list(data0.columns))
#find out number of categories for each variable
#print(data0.job.unique())
#print(data0.job.value_counts())

# exploring data
feature_var=['job','marital','education','default','housing','loan','month','day_of_week','poutcome','y']
for var in feature_var:
    x=data0[var].value_counts()
    print(x)

# education 
data0.education=np.where(data0.education=='basic.9y','basic',data0.education)
data0.education=np.where(data0.education=='basic.6y','basic',data0.education)
data0.education=np.where(data0.education=='basic.4y','basic',data0.education)

print(data0.education.value_counts())

# visualization of data - dependent variable y
sns.countplot(x='y',data=data0, palette='hls')
print(data0.y.value_counts())

# visualization of dependent variables 

features_var=['job','marital','education','default','housing','loan','month','day_of_week','poutcome']

for var in features_var:
    plt.figure(figsize=(12,12))
    table_var0=pd.crosstab(data0[var],data0['y'])
    table_var0.div(table_var0.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
    title1='Purchase Frequency for' + ' ' + var
    plt.title(title1)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.legend(loc='best')

# based on the plots generated these are the varibale that correlate with y(could be a potential predictor for y):
    # Job title, education,default, month, poutcome
# varibale that doesn't seems to be a go predictor for y are
    # day_of _week, loan, housing, marital
plt.figure(figsize=(12,12))    
plt.hist(data0['age'])
plt.xlabel('age')
plt.ylabel('Frequency')
plt.legend(loc='best')

## replacing all the variable with dummy variable
#
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    g =var + '_dummy' 
    g = pd.get_dummies(data0[var],prefix=var) 
    data0.drop(var,axis=1,inplace=True)
    data1=data0.join(g)
    data0=data1
##       
# generate heatmap with the correlation of data
data_correlation_matrix=data0.corr()
plt.figure(figsize=(12,12))
sns.heatmap(data_correlation_matrix,center=0,cmap='RdBu_r')
    
#define indepenet variable variable
X=data0.iloc[:,data0.columns!='y']
y=data0.iloc[:,data0.columns=='y']

#split data into train data and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# creat a LogisticRegression model
LogReg_Mod = LogisticRegression()
LogReg_Mod.fit(X_train,y_train)
y_pred=LogReg_Mod.predict(X_test)
yy=LogReg_Mod.predict_proba(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('the coefs of logistic regression model are:', LogReg_Mod.coef_)
print('The accuracy of prediction is ', accuracy_score(y_test,y_pred))


Logit_roc_auc = roc_auc_score (y_test,y_pred)
print('The area under the ROC curve is:', Logit_roc_auc)
#
fpr,tpr,thresholds = roc_curve(y_test,yy[:,1])

plt.figure(figsize=(4,4))
plt.plot(fpr,tpr,'r--')
plt.plot([0,1],[0,1],'k-')
plt.xlabel('Faulse Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')


    
    




