# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 00:24:24 2021

@author: H Ni
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
# 数据库连接信息，该数据库保存了近两年的股票数据
con = create_engine('mysql+pymysql://study:524524524@47.93.28.75/test')

def set_portfolio(codes, con = con):
  ret_list = pd.DataFrame()
  for code in codes:
    # 循环读入股票信息
    dat = pd.read_sql_table(code,con, columns = ['pct_chg'], index_col = 'trade_date')
    dat.rename(columns = {'pct_chg':code}, inplace=True) #将列标签换成股票的ts_code
    # 合并不同股票的日收益
    ret_list = pd.concat([ret_list, dat], axis = 1 ,join = 'outer', ignore_index = False, sort = False)
  # 去掉空值行
  ret_list.dropna(axis = 0, how = 'any', inplace = True)
  return ret_list


codes = ['000002','000100','000776','600030','600177']
#codes = ['000002','399300']
ret_list = set_portfolio(codes) 

from scipy import stats
def multi_corr(ret_list_pair, winlen = 50, N = 100):
    print('Correlation coefficent between '+ret_list_pair.columns.values[0]+' and '+ret_list_pair.columns.values[1]+' is: ' + str(np.round(np.corrcoef(ret_list_pair.T)[0,1], 3)))
    corr_list = []
    for i in range(N):
        start = np.random.randint(len(ret_list_pair) - winlen)
        corr_list.append(np.corrcoef(ret_list_pair.iloc[start:start+winlen,0],ret_list_pair.iloc[start:start+winlen,1])[0,1])        
    print('Average correlation is: '+str(round(np.mean(corr_list),3))+' with std= '+ str(round(np.std(corr_list),3)))  
    # Display
    plt.figure()
    plt.hist(corr_list, bins=20)
    plt.show()  
    print('Kurtosis is: '+ str(round(stats.kurtosis(corr_list),3)) + '; Shewness is ' + str(round(stats.skew(corr_list),3)))
    
multi_corr(ret_list)


def port_stat(ret_list, num_assets, num_ports, short_sell = 0):

    weights = np.random.random([num_ports,num_assets]) if short_sell == 0 else np.random.randn(num_ports,num_assets) #根据组合数和投资的资产数建立一个矩阵
    #np.random.random([ ，])生成的是0～1之间的小数；np.random.randn( ， )生成的是服从标准正态的数。
    weights = weights / (np.mat(np.sum(weights, axis = 1)).T* np.ones([1,num_assets]))
    cov_matrix = np.cov(ret_list.T)
    
    risk_matrix = weights * cov_matrix * weights.T
    ports_risks = np.sqrt(np.diag(risk_matrix)) #diag函数返回对角线元素
    ports_risks = ports_risks.reshape(-1,1) #重塑成一列
    ports_returns = np.array(weights * np.mat(np.mean(ret_list.values, axis = 0)).T)
    
    return ports_risks, ports_returns, cov_matrix, weights

# calculate portfolio with 2 assets
num_ports = 100 #组合的数量
n = 2
ports_risks, ports_returns, cov_matrix, weights = port_stat(ret_list.iloc[:,:n], n, num_ports)

fig, ax = plt.subplots()
ax.scatter(ports_risks, ports_returns, alpha=0.4) #资产随机配置时的风险与收益率
ax.scatter(np.sqrt(np.diag(cov_matrix)), np.mean(ret_list.iloc[:,:n].values, axis = 0), c = '#FF0000', alpha=0.4) #全部买这个资产的风险与收益率
ax.set_title('The risk and return of the selected asset')
ax.set_xlabel('$\sigma_p$')
ax.set_ylabel('$R_p$')


# calculate portfolio with 3 or more assets
num_ports = 1000
n = 3
ports_risks, ports_returns, cov_matrix, weights = port_stat(ret_list.iloc[:,:n], n, num_ports)

fig, ax = plt.subplots()
ax.scatter(ports_risks, ports_returns, alpha=0.4) #资产随机配置时的风险与收益率
ax.scatter(np.sqrt(np.diag(cov_matrix)), np.mean(ret_list.iloc[:,:n].values, axis = 0), c = '#FF0000', alpha=0.4) #全部买这个资产的风险与收益率
ax.set_title('The risk and return of the selected asset')
ax.set_xlabel('$\sigma_p$')
ax.set_ylabel('$R_p$')




cs = pd.read_sql_table('cis300',con, columns = ['con_code'])
codes = [c.split('.')[0] for c in list(cs['con_code'])]
random_select = list(set(random.choices(codes, k = 50)))
ret_list = set_portfolio(random_select)

p_risk = []
for i in range(len(random_select)):
  n = i + 1
  p_risk.append(np.dot(np.dot(1/n * np.ones([1,n]), ret_list.iloc[:,:n].cov().values), np.ones([n,1]) * 1/n))

# 作图
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(random_select)),np.squeeze(p_risk))
ax.set_title('Average Portfolio Risk')
ax.set_xlabel('Number of Stocks in Portfolio')
ax.set_ylabel('Standard Deviation(%)')


# Average diversification
plt.figure()
plt.hist(ports_risks, bins = 20)
plt.show()
print('The average risks of components stocks is: ' + str(np.mean(np.sqrt(np.diag(cov_matrix)))))
print('The average risks of portfolios is: ' + str(np.mean(ports_risks)))


def sharpe(ports_risks, ports_returns, risk_free_rate):  #计算夏普比率
    return (ports_returns - risk_free_rate) / ports_risks

risk_free_rate = 0.02
ind_max_sharpe = np.argmax(sharpe(ports_risks, ports_returns, risk_free_rate)) #找出夏普比率最大的配置情况

plt.figure()
plt.scatter(ports_risks, ports_returns, alpha=0.4)
plt.scatter(np.sqrt(np.diag(cov_matrix)), np.mean(ret_list.iloc[:,:n].values, axis = 0), c = 'darkblue', alpha= 1 )
plt.scatter([0,ports_risks[ind_max_sharpe]], [risk_free_rate, ports_returns[ind_max_sharpe]], alpha=1, c = 'red') #画出使得夏普比率最大的资产配置线
plt.plot(np.array([0,ports_risks[ind_max_sharpe]]),np.array([risk_free_rate, ports_returns[ind_max_sharpe]]), alpha=0.3, c = 'red')


def utility_func(cports_risks, cports_returns, aversion): #构建效用函数
    return cports_returns - 0.005 * aversion * cports_risks * cports_risks

aversion = 6
num = 50
cports_risks = np.linspace(0,ports_risks[ind_max_sharpe], num = num) 
cports_returns = np.linspace(risk_free_rate,ports_returns[ind_max_sharpe], num = num)
index_comp_port = np.argmax(utility_func(cports_risks, cports_returns, aversion))

plt.figure()
# opportunity sets
plt.scatter(ports_risks, ports_returns, alpha=0.4)
# original assets
plt.scatter(np.sqrt(np.diag(cov_matrix)), np.mean(ret_list.iloc[:,:n].values, axis = 0), c = 'darkblue', alpha= 1 )
# Risk free asset and max sharpe portfolio
plt.scatter([0,ports_risks[ind_max_sharpe]], [risk_free_rate, ports_returns[ind_max_sharpe]], alpha= 0.5, c = 'orange')
# Capital allocation line
plt.plot(np.array([0,ports_risks[ind_max_sharpe]]),np.array([risk_free_rate, ports_returns[ind_max_sharpe]]), alpha= 0.3, c = 'red')
# Complete optimal portfolio
plt.scatter(cports_risks[index_comp_port], cports_returns[index_comp_port], alpha= 1, c = 'darkred') #在资产配置线上找到使得效用最大的点

# Optimal weights

weights_on_risk_free = 1 - index_comp_port / num
weights_on_risky = (index_comp_port / num) * np.array(weights[ind_max_sharpe,:])

optimal_weights = [weights_on_risk_free, weights_on_risky.tolist()[0]]

print('Invest {:.5f}% on risk free assets'.format(100*np.round(weights_on_risk_free,5)))

for i in range(n):
    print('Invest {0}% on Stock {1}'.format(100*np.round(weights_on_risky.tolist()[0][i],5),codes[i]))
    
    

