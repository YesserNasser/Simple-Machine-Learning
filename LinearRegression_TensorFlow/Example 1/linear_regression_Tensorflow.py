# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:13:04 2019
@author: Yesser H. Nasser
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data():
    x1_data = np.linspace(0,2,100)
    x2_data = np.linspace(-1,1,100)
    y_data = 2*x1_data + 1.5*x2_data + np.random.randn(*x1_data.shape)*0.2 + 0.7
    return x1_data,x2_data,y_data

# constract tensorflow
# placeholders
def linear_regression():
    x1 = tf.placeholder(tf.float32, shape=(None,), name = 'x1')
    x2 = tf.placeholder(tf.float32, shape=(None,), name = 'x2')
    y = tf.placeholder(tf.float32, shape=(None,), name = 'y')
    
    # variables
    with tf.variable_scope('lreg') as scope:
        W1 = tf.Variable(np.random.normal(), name = 'W1')
        W2 = tf.Variable(np.random.normal(), name = 'W1')
        b = tf.Variable(np.random.normal(), name = 'b')
        
        y_pred = W1*x1 + W2*x2 + b
        # loss function
        loss = tf.reduce_mean(tf.square(y_pred - y))
        
    return x1,x2,y,y_pred,loss

x1_data,x2_data,y_data = generate_data()
x1,x2,y,y_pred,loss = linear_regression()

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.04).minimize(loss)

# run the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {x1: x1_data, x2: x2_data, y: y_data}
    
    for i in range (200):
        sess.run(optimizer,feed_dict)
        print(i,'loss', loss.eval(feed_dict))
        plt.scatter(i,loss.eval(feed_dict),c='r', marker = '*')

    y_pred_synth = sess.run(y_pred, {x1: x1_data, x2: x2_data})
    
    plt.grid(1)
    plt.xlim(0,100)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')

fig=plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x1_data,x2_data,y_data)
ax.plot(x1_data,x2_data,y_pred_synth, c='r', marker='*')

ax.set_xlabel('x1_data')
ax.set_ylabel('x2_data')
ax.set_zlabel('y_data')
