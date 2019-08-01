# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:57:24 2018
@author: Yesser H. Nasser
"""
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

# loading the data
x = load_boston()
df = pd.DataFrame(x.data, columns=x.feature_names)
df['MEDV'] = x.target
X = df.drop('MEDV', 1)
y = df['MEDV']
df.head()

'''
Select the method to work with:
    0: Backward Elimination
    1: Recursive Feature Elimination
    2: Embadded Method without threshold
    3: Embadded Method with threshold
'''
Feature_Selection_Method = 3


if Feature_Selection_Method == 0:
    '''========================================================================='''
    '''======================== Wrapper Methods ================================'''
    '''========================================================================='''
    ''' ======================== Backward Elimination ========================= '''
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index = cols)
        pmax = max(p)
        feature_with_max_p = p.idxmax()
        if (pmax>0.05):
            print(feature_with_max_p, pmax)
            cols.remove(feature_with_max_p)
        else:
            break
    selected_features_BE = cols
    print('Based on the Backward Elimination method the selected features are: ', selected_features_BE)
    
elif Feature_Selection_Method == 1:     
    ''' ==================== Recursive Feature Elimination ==================== '''
    nof_list = np.arange(1, len(X.columns))
    high_score = 0
    nof =0
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
        model = LinearRegression()
        rfe = RFE(model,nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe, y_train)
        score = model.score(X_test_rfe,y_test)
        if (score>high_score):
            high_score = score
            nof = nof_list[n]
    print('Optimum number of features %d' %nof)
    print('Score with %d features is %f' %(nof, high_score))
    
    model = LinearRegression()
    rfe = RFE(model,nof)
    
    X_rfe = rfe.fit_transform(X,y)
    model.fit(X_rfe, y)
    temp = pd.Series(rfe.support_,index = X.columns)
    selected_features_RFE = list(temp[temp==True].index)
    print('Based on the Recursive Feature Elimination method the selected features are: ',selected_features_RFE)
    
    
elif Feature_Selection_Method == 2:   
    '''========================================================================='''
    '''=========== Embedded Method (Lasso) without threshold ==================='''
    '''========================================================================='''
    # Lasso without threshold
    reg = LassoCV(cv=5)
    reg.fit(X,y)
    print('Best Alpha using build-in Lasso: %f' %reg.alpha_)
    print('Best Score using build-in Lasso: %f' %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    selected_features_lassoCV = list(coef[coef!=0].index)
    print('Based on the Embedded method (Lasso) (without threshold) the selected features are: ', selected_features_lassoCV)
    imp_coef=coef.sort_values()
    plt.figure(figsize=(6,4))
    imp_coef.plot(kind='barh')
    plt.title('Feature importance using Lasso Model')
    plt.grid(0.25)
    print('Lasso model picked ' + str(sum(coef!=0)) + ' features' + ' and removed ' + str(sum(coef==0)) + ' features')
    
    
elif Feature_Selection_Method == 3:   
    '''========================================================================='''
    '''============== Embedded Method (Lasso) with threshold ==================='''
    '''========================================================================='''
    reg = LassoCV(cv=5)
    reg.fit(X,y)
    # Lasso with threshhold to select the importnat features
    sfm = SelectFromModel(reg, threshold = 0.2)
    sfm.fit(X,y)
    n_features = sfm.transform(X).shape[1]
    print ('The  model selected %d' %n_features + ' important features' + ' and removed %d' % (X.shape[1]-n_features) + ' less important features')
    Features = pd.Series(sfm.get_support(), index = X.columns)
    print(Features)
    Selected_Features_sfm = list(Features[Features == True].index)
    print('Based on the Embedded method (Lasso) (with threshold of 0.2) the selected features are: ', Selected_Features_sfm)
    
else:
    print('Please enter 0: for BE, 1: for RFE, 2: for Embedded Method without threshold, 3: for Embedded Method with threshold')
