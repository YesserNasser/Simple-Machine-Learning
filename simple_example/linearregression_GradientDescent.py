# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:46:39 2019

@author: Yesser
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
data=pd.read_csv('ex1data1.txt', names=['population','profit'])

#defining X and Y
X_df=pd.DataFrame(data.population)
#X=data.iloc[:,:-1].values
Y_df=data.iloc[:,1].values

m=len(X_df)

#ploting the data
fig=plt.figure(figsize=(12,6))
plt.scatter(X_df,Y_df,c='red',marker='x')
plt.xlabel('Population of city in 10,000s', fontsize=15)
plt.ylabel('Profit in $10,000s', fontsize=15)
plt.xlim([5,25])
plt.ylim([-5,25])
plt.grid()

# =============================================================================
#  The idea of linear regression is to find a relationship between our target or dependent variable (Y) and a set of explanatory variables (X1,X2,X3....). 
#  This relatonship can then be used to predict other values.
#  In our case with one variable, this relationship is a line defined by parameters B and the following form: Y = B0 + B*X , where B0 is our intercept.
#  This can be extended to multivariable regression by extending the equation in vector form: Y = X*B 
#  So how do I make the best line? In this figure, there are many possible lines. Which one is the best?
# =============================================================================

plt.figure(figsize=(10,8))
plt.scatter(X_df,Y_df, c='k',marker='.')
plt.plot([5,22],[6,6], '-')
plt.plot([5,22],[0,20], '-')
plt.plot([5,15],[-5,25], '-')

# =============================================================================
# =============================================================================

x_quad=[n/10 for n in range(0,100)]
y_quad=[(n-4)**2 + 5 for n in x_quad]

plt.figure(figsize=(10,7))
plt.plot(x_quad,y_quad,'k--')
plt.axis([0,10,0,30])
plt.plot([1,2,3],[14,9,6], 'ro')
plt.plot([5,7,8],[6,14,21], 'bo')
plt.plot(4, 5, 'ko')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Quadratic Equation')

# =============================================================================
# =============================================================================

iterations=1500
alpha=0.01
# add a columns of 1s as intercept to X
X_df['intercept']=1
## transform to numpy arrays for easier matrix math and start theta at 0
X=np.array(X_df)
Y = np.array(Y_df).flatten()

theta = np.array([0,0])
#==============================================================================
def cost_function(X, Y, theta):
     #number of training examples
     m=len(Y)
     #calculate the cost with the given parameters
     J=np.sum((X.dot(theta)-Y)**2)/2/m
     
     return J
 #=============================================================================


def gradient_descent(X, y, theta, alpha, iterations):
    """
    gradient_descent Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    cost_history = [0] * iterations
    
    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis-y
        gradient = X.T.dot(loss)/m
        theta = theta - alpha*gradient
        cost = cost_function(X, y, theta)
        cost_history[iteration] = cost

    return theta, cost_history


(t, c) = gradient_descent(X,Y,theta,alpha, iterations)
print(t)

print (np.array([3.5,1]).dot(t))
print (np.array([7,1]).dot(t))

best_fit_x = np.linspace(0, 25, 20)
best_fit_y = [t[1] + t[0]*xx for xx in best_fit_x]

plt.figure(figsize=(10,6))
plt.plot(X_df.population, Y_df, '.')
plt.plot(best_fit_x, best_fit_y, '-')
plt.axis([0,25,-5,25])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs. Population with Linear Regression Line')