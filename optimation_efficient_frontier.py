# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:27:57 2020

@author: Yusufdexter
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.optimize import minimize




ticker = ['AAPL', 'CSCO', 'IBM', 'AMZN']

aapl = wb.DataReader(ticker[0], data_source = 'yahoo', start = '2015-1-1')['Adj Close']
cisco = wb.DataReader(ticker[1], data_source = 'yahoo', start = '2015-1-1')['Adj Close']
ibm = wb.DataReader(ticker[2], data_source = 'yahoo', start = '2015-1-1')['Adj Close']
amzn = wb.DataReader(ticker[3], data_source = 'yahoo', start = '2015-1-1')['Adj Close']

aapl = pd.DataFrame(aapl)
cisco = pd.DataFrame(cisco)
ibm = pd.DataFrame(ibm)
amzn = pd.DataFrame(amzn)

# Concatenating
stocks = pd.concat([aapl, cisco, ibm, amzn], axis = 1)

# Renaming the columns
stocks.columns = ['AAPL', 'CISCO', 'IBM', 'AMZN']


# DAILY RETURNS
daily_returns = stocks.pct_change(1)

# Visualisation
daily_returns.hist(bins = 100, figsize = (8, 8))
plt.savefig('0.1 stock daily return.png')

# Average daily returns
avg_daily = daily_returns.mean()
avg_daily

# Correlation between the stocks
daily_returns.corr()

# Co-Variance between the stocks
daily_returns.cov()



'''Single Run for Some Random Allocation'''
# Set seed (optional)
np.random.seed(101)

num_portfolios = 15000

all_weights = np.zeros((num_portfolios,len(stocks.columns)))
return_arr = np.zeros(num_portfolios)
volatility_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

for ind in range(num_portfolios):
    # Create Random Weights
    weights = np.array(np.random.random(4))
    # Rebalance Weights
    weights = weights / np.sum(weights)
    # Save Weights
    all_weights[ind,:] = weights
    # Expected Return
    return_arr[ind] = np.sum((daily_returns.mean() * weights) * 252)
    # Expected Variance
    volatility_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
    # Sharpe Ratio
    sharpe_arr[ind] = return_arr[ind] / volatility_arr[ind]



# OPTIMUM Value
sharpe_arr.max()

# OPTIMUM Location
sharpe_arr.argmax()

# OPTIMUM ALLOCATION
print(stocks.columns)
all_weights[12242,:]


max_sr_return = return_arr[12242]
max_sr_volatility = volatility_arr[12242]

# VISUALISATION
plt.figure(figsize = (16, 8))
plt.scatter(volatility_arr, return_arr, c = sharpe_arr)
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
# Add black dot for max SHARPE RATIO
plt.scatter(max_sr_volatility, max_sr_return, c = 'black', s = 50, edgecolors = 'black')
plt.title('(SHARPE RATIO) \nTHE OPTIMUM PORTFOLIO ALLOCATION')
plt.annotate('The black dot is the Most Optimum Stock Combination', 
             (0, 0), (50, -30), fontsize = 10, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.savefig('0.2 Sharpe Ratio.png')


'''MATHEMATICAL OPTIMISATION'''

def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return, volatility, sharpe ratio
    """
    weights = np.array(weights)
    returns = np.sum(daily_returns.mean() * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
    sr = returns / volatility
    return np.array([returns, volatility, sr])


def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * - 1


# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1

# By convention of minimize function it should be a function that returns zero for conditions
cons = ({'type':'eq','fun': check_sum})

# 0-1 bounds for each weight
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

# Initial Guess (equal distribution)
init_guess = [0.25, 0.25, 0.25, 0.25]

# Sequential Least SQuares Programming (SLSQP).
opt_results = minimize(neg_sharpe, init_guess, method = 'SLSQP', bounds = bounds, constraints = cons)
opt_results

opt_results.x

get_ret_vol_sr(opt_results.x)


'''
    All Optimal Portfolios (Efficient Frontier)
'''

# Our returns go from 0 to somewhere along 0.3
# Create a linspace number of points to calculate x on
frontier_y = np.linspace(0, 0.4, 100) # Change 100 to a lower number for slower computers!

def minimize_volatility(weights):
    return  get_ret_vol_sr(weights)[1] 

frontier_volatility = []
for possible_return in frontier_y:
    # function for return
    cons = ({'type':'eq','fun': check_sum},
            {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = minimize(minimize_volatility, init_guess, method = 'SLSQP', bounds = bounds, constraints = cons)
    
    frontier_volatility.append(result['fun'])


# VISUALIZATION
plt.figure(figsize = (16, 8))
plt.scatter(volatility_arr, return_arr, c = sharpe_arr)
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('The Efficient Frontier')
# Add frontier line
plt.plot(frontier_volatility, frontier_y, 'g--', linewidth = 3)
plt.annotate('Stock Combination along the curve is Efficient', 
             (0, 0), (50, -30), fontsize = 10, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.savefig('0.3 Efficient Frontier.png')
