"""
Created on Wed Sep 16 22:12 2020

@author: Liangying Liu
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import KFold
from sklearn.model_selection import permutation_test_score

def get_sample_balcv(x, y, nfolds,pthresh):
    """
    This function uses anova across CV folds to find
    a set of folds that are balanced in their distriutions
    of the X and Y values - see Kohavi, 1995
    """
    nsubs = len(y)
    #cv = KFold(n_splits = nfolds, shuffle=True)

    # cycle through until we find a split that is good enough

    good_split = 0
    while good_split == 0:
        fold = KFold(n_splits=nfolds, shuffle=True)
        ctr = 0
        idx = np.zeros((nsubs, nfolds))  # this is the design matrix
        for train, test in fold.split(x):
            idx[test, ctr] = 1
            ctr += 1

        lm_x = OLS(x - np.mean(x), idx).fit()
        lm_y = OLS(y - np.mean(y), idx).fit()

        if lm_x.f_pvalue > pthresh and lm_y.f_pvalue > pthresh:
            good_split = 1

    # do some reshaping needed for the sklearn linear regression function
    x = x.reshape((nsubs, 1))     # sklearn的格式：行为数据点（被试数目），列为数据的维度
    y = y.reshape((nsubs, 1))

    pred = np.zeros((nsubs, 1))

    for train, test in fold.split(x):
        lr = LinearRegression()
        lr.fit(x[train, :], y[train, :])
        pred[test] = lr.predict(x[test])

    return np.corrcoef(pred[:, 0], y[:, 0])[0, 1]

def cv_rep(x,y,nfolds,nruns,pthresh):
    corrs = np.zeros(nruns)

    for run in range(nruns):
        corrs[run] = get_sample_balcv(x, y, nfolds,pthresh)

    corr_mean = np.mean(corrs)

    return corr_mean


def perm_test(x,y,nperm,nfolds,nruns,pthresh):
    list = np.empty(nperm)
    r = cv_rep(x,y,nfolds, nruns,pthresh)
    k = 0
    for i in range(nperm-1):
        np.random.shuffle(y)
        list[i] = cv_rep(x,y,nfolds, nruns,pthresh)
        k += r < list[i]
    p = k / (nperm-1)
    return r,p



data = pd.read_csv (r'C:\Users\liuxiaomiao\Desktop\haiyang\data_final_NC\FPN_v_anxiety.csv')
#data = pd.read_csv(r'C:\Users\liuxiaomiao\Desktop\haiyang\data_final_NC\Prediction analysis\SCL_trait.csv')

#data = pd.read_csv(r'C:\Users\liuxiaomiao\Desktop\haiyang\data_final_NC\meanRT_hit.csv')
#data = pd.read_csv(r'C:\Users\liuxiaomiao\Desktop\haiyang\data_final_NC\Prediction analysis\IPS_trait_stress_mean.csv')

'''
RT_stress_2back = data[(data['group'] == 0) & (data['cond'] == 2)]['rt']
v_stress_2back = data[(data['group'] == 0) & (data['cond'] == 2)]['v']

RT_control_2back = data[(data['group'] == 1) & (data['cond'] == 2)]['rt']
v_control_2back = data[(data['group'] == 1) & (data['cond'] == 2)]['v']


x = RT_control_2back
y = v_control_2back

x1 = data[(data['group'] == 0)]['x1']
x2 = data[(data['group'] == 0)]['x2']
y =  data[(data['group'] == 0)]['y']

x1 = x1.values	#converting it into an array
x2 = x2.values
y = y.values

x = np.vstack((x1,x2))'''

x = data[(data['group'] == 0)]['MFG']
y = data[(data['group'] == 0)]['v']

x = x.values
y = y.values

nfolds = 4
nruns =  4
nperm = 500
pthresh = 0.5

r,p = perm_test(x,y,nperm,nfolds,nruns,pthresh)
print(r,p)



