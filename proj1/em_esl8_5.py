# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:29:45 2019

@author: daksh
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.asarray([[-0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53, 0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22]])

idx1 = np.random.choice(data.shape[1], 1, replace=False)
idx2 = np.random.choice(data.shape[1], 1, replace=False)

mu1 = data[0,idx1]
mu2 = data[0,idx2]

#sig1 = st.variance(data[0,:])
#sig2 = st.variance(data[0,:])
var1 = np.sum(np.square(data[0,:] - np.mean(data[0,:])))/20
var2 = np.sum(np.square(data[0,:] - np.mean(data[0,:])))/20

pi_hat = 0.5
pi_hat_ar = np.array([])

def gauss_fun(x, mu, var):
    pd = (1/(np.sqrt(2*np.pi*var)))*np.exp(-1*np.square(x-mu)/(2*var))
    return pd

for ii in range(0,50):
    gamma_hat_ar = np.array([])
    for i in range(0,20):
        # Expectation Step:
        gamma_hat = (pi_hat*gauss_fun(data[0,i], mu2, var2))/((1-pi_hat)*gauss_fun(data[0,i], mu1, var1) + pi_hat*gauss_fun(data[0,i], mu2, var2))
        gamma_hat_ar = np.append(gamma_hat_ar, gamma_hat)
    # Maximization Step:
    mu1 = np.sum(np.multiply(1-gamma_hat_ar, data[0,:]))/np.sum(1-gamma_hat_ar)
    mu2 = np.sum(np.multiply(gamma_hat_ar, data[0,:]))/np.sum(gamma_hat_ar)

    var1 = np.sum(np.multiply(1-gamma_hat_ar, np.square(data[0,:]-mu1)))/np.sum(1-gamma_hat_ar)
    var2 = np.sum(np.multiply(gamma_hat_ar, np.square(data[0,:]-mu2)))/np.sum(1-gamma_hat_ar)

    pi_hat = np.sum(gamma_hat_ar/20)
    pi_hat_ar = np.append(pi_hat_ar, pi_hat)

    #like = np.sum(np.log())

pd_arr = np.array([])

for i in range(0,20):
    pd1 = gauss_fun(data[0,i], mu1, var1)
    pd2 = gauss_fun(data[0,i], mu2, var2)
    pd = (1-pi_hat)*pd1 + pi_hat*pd2
    pd_arr = np.append(pd_arr, pd)

plt.plot(data[0,:], pd_arr, 'r.')
plt.show()
