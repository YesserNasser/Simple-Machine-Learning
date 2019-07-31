# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:41:59 2018

@author: Yesser
"""
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
sns.set_style('whitegrid')

# defining the historical percent to target performance 
# historically the perfomance is normal distibution with mean =100% standard deviation of 10% 
# number of sales representative is 500
# umber of simulation is 1000 times. 
avg=1
std_dev=0.1
num_reps=50
num_simulation=1000

sales_target_values=[75000,100000,200000,300000,400000,500000]
sales_target_prob=[0.3,0.3,0.2,0.1,0.05,0.05]

# Define a list to keep all the results from each simulation that we want to analyze 
all_stats=[]

# loop through many simulation
for i  in range (num_simulation):
    # choose random input for the sales targets and percent to target
    pct_to_target=np.random.normal(avg,std_dev,num_reps).round(2)
    sales_target=np.random.choice(sales_target_values,num_reps,p=sales_target_prob)  
    #creating data data frame
    df=pd.DataFrame(index=range(num_reps),data={'Sales_Target':sales_target,'Pct_To_Target':pct_to_target})
    # claculate sales
    df['Sales']=df['Pct_To_Target']*df['Sales_Target']
    #defining a function to calculate the comission
    def cal_commission_rate(x):
        if x<=0.9:
            return 0.02
        if x<=0.99:
            return 0.03
        else:
            return 0.04
    # create commission rate columns to dataframe
    df['Commission_Rate']=df['Pct_To_Target'].apply(cal_commission_rate)
    df['Commission_Amount']=df['Sales']*df['Commission_Rate']
    
    # we want to track sales, commission amounts and sales trgets over all the simulation
    all_stats.append([df['Sales'].sum().round(0),df['Commission_Amount'].sum().round(0),df['Sales_Target'].sum().round(0)])
    
    
results_df=pd.DataFrame.from_records(all_stats, columns=['Sales','Commission_Amount','Sales_Target'])
#results_df.describe().style.format('{:,}')

fig=plt.figure(figsize=(8,8))   
plt.hist(results_df['Commission_Amount'])
plt.xlabel('Total_Commission_Amount', fontsize=15)
plt.ylabel('Frequency', fontsize=15)    
plt.title('Historgram Total Commission Using Monte Carlo Simulation 10000', fontsize=15)

