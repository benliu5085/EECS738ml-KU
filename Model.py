# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:31:56 CST 2019

@author: ben
"""

import pandas
import sys
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from scipy.stats import norm

"""********************** Macro definition **********************************"""
NUMBER_OF_FEATURE = 1               # number of feature that we are going to use
FEATURE_TYPE = ['float']            # types of feature that we are going to choose
BATCH_BOUND = 13                    # minimal number of data points for a cluster
MAXIMAL_REPEAT = 100                # maximal times of iterations
EPISILON = 1e-12                    # a small number to determine vector equal
EPISILON1 = 1e-5                    # a small number to determine vector equal
LEVEL_OF_CONFIDENCE = 3             # 99 % data points
"""**************************************************************************"""

def dist(example, cc):
    KC = cc[0]
    for i in range(1, len(cc)):
        KC = np.append(KC, cc[i], 0)
    for i in range(0, len(KC)):
        KC[i] = KC[i] - example
    return (KC*KC.transpose()).diagonal()

def kmCluster(data_in, K):
    if K == 1:
        return [data_in]
    centers = random.sample(data_in, K)
    for cnt_repeat in range(0, MAXIMAL_REPEAT):
        ff_restart = False
        clusters = copy.copy(centers)
        for i in range(0, len(data_in)):
            t = dist(data_in[i], centers)
            clr_id = (t.argmin(1))[0,0]
            clusters[clr_id] = np.append(clusters[clr_id], data_in[i], 0)
        for i in range(0, len(centers)):
            clusters[i] = np.delete(clusters[i], 0, 0) # remove the repeated center from each cluster
        new_centers = []
        for cm in clusters:
            if len(cm) == 0:
                # restart when there is an empty cluster
                new_centers = random.sample(data_in, K)
                ff_restart = True
                break
            else:
                new_centers.append(cm.mean(0))
        if not ff_restart:
            move_dist = 0
            for i in range(0, len(centers)):
                move_dist += ((centers[i]-new_centers[i])*(centers[i]-new_centers[i]).transpose())[0,0]
            if move_dist < EPISILON: # quick stop condition : didn't move much
                break
        centers = copy.copy(new_centers)
    return clusters

def multi_gauss_fun(Vx, Vmu, Vsigma):
    return (1/(np.sqrt(np.power((2*np.pi),len(Vsigma))*np.linalg.det(Vsigma))))*np.exp(-0.5*((Vx-Vmu)*(Vsigma.getI())*((Vx-Vmu).transpose()))[0,0])


"""************************ Usage *******************************************"""
print("call by\n    Model.py <file.csv> [<int.numer_of_features>]")
print("Note: <int.numer_of_features> is 1 by default for better visulization")
print("      of modeling mixture of Gaussians, set it to be 2 to visulize")
print("      K-means clustering. There will not be any visulization for ")
print("      other value, instead the final parameters, Pi, Mu and Sigma.\n\n")
"""**************************************************************************"""

if len(sys.argv) == 2:
    fname = sys.argv[1]
elif len(sys.argv) == 3:
    fname = sys.argv[1]
    NUMBER_OF_FEATURE = int(sys.argv[2])
else:
    exit()

df = pandas.read_csv(fname)
df = df.dropna(0)
exclude = list(df.columns)[-1]  # label
df1 = df.select_dtypes(include=FEATURE_TYPE)
pool = list(df1.columns)
if exclude in pool:
    pool.remove(exclude)

"""1: pick NUMBER_OF_FEATURE numeric feature and normalize them"""
# pick feature
chosen_col = []
if len(pool) <= NUMBER_OF_FEATURE:
    chosen_col = pool
else:
    chosen_col = random.sample(pool, NUMBER_OF_FEATURE)

# normalization
raw_data = np.matrix(df1.loc[:,chosen_col])
minima = raw_data.min(0)
maxima = raw_data.max(0)
a = np.zeros((NUMBER_OF_FEATURE,NUMBER_OF_FEATURE))
np.fill_diagonal(a, maxima-minima)
K = np.matrix(a)
K = K.getI()
B = np.tile(minima, (len(raw_data),1))
data = (raw_data - B) * K

"""2: cluster by K-means, determine K automatically """
print(
"----------step 1: clustering using K-Means------------------------------------\n")
cnt_cluster = 1
while cnt_cluster < len(data)/BATCH_BOUND:
    print("trying " + str(cnt_cluster) + " cluster...")
    temp_cluster = kmCluster(data, cnt_cluster)
    cnt_outliers = 0
    for cm in temp_cluster:
        this_c = cm.mean(0)
        this_std = cm.std(0, ddof = 1)
        for pp in cm:
            diff_v = pp - this_c
            if (diff_v*(diff_v.transpose()))[0,0] > (LEVEL_OF_CONFIDENCE*LEVEL_OF_CONFIDENCE*(this_std*(this_std.transpose()))[0,0]):
                cnt_outliers += 1
            if cnt_outliers > BATCH_BOUND:
                break
        if cnt_outliers > BATCH_BOUND:
            break

    print(str(cnt_outliers) + " outliers\n")
    if cnt_outliers > BATCH_BOUND:
        cnt_cluster += 1
    else:
        break

"""3: Model parameter using EM algorithm """
# could vectorization to speed up
# initialization
# xi in data[i]
# mu_k in Mu[k]
# sigma_k in Sigma[k]
print("\n\n")
print(
"----------step 2: Modelling mixture of Gaussians using EM----------------------\n")

Mu = [0] * len(temp_cluster)
Sigma = [0] * len(temp_cluster)
Pi = [0] * len(temp_cluster)
for i in range(0, len(temp_cluster)):
    Mu[i] = temp_cluster[i].mean(0)
    Sigma[i] = np.matrix(np.cov(temp_cluster[i], rowvar=False))
    Pi[i] = float(len(temp_cluster[i])) / float(len(data))

Gamma = np.zeros([len(data), len(temp_cluster)])

for cnt_repeat in range(0, MAXIMAL_REPEAT):
    # E-step
    for i in range(0, len(data)):
        for k in range(0, len(temp_cluster)):
            Gamma[i,k] = Pi[k] * multi_gauss_fun(data[i], Mu[k], Sigma[k])
    SG = Gamma.sum(1)
    for i in range(0, len(data)):
        for k in range(0, len(temp_cluster)):
            Gamma[i,k] = Gamma[i,k] / SG[i]

    # M-step
    NK = Gamma.sum(0)

    New_Mu = [0] * len(temp_cluster)
    for k in range(0, len(temp_cluster)):
        temp = Gamma[0,k] * data[0]
        for i in range(1, len(data)):
            temp += Gamma[i,k] * data[i]
        New_Mu[k] = temp / NK[k]

    New_Pi = list(NK / len(data))

    New_Sigma = [0] * len(temp_cluster)
    for k in range(0, len(temp_cluster)):
        temp = Gamma[0,k] * (data[0]-New_Mu[k]).transpose() * (data[0]-New_Mu[k])
        for i in range(1, len(data)):
            temp += Gamma[i,k] * (data[i]-New_Mu[k]).transpose() * (data[i]-New_Mu[k])
        New_Sigma[k] = temp / NK[k]

    # convergence
    div = 0
    for k in range(0, len(temp_cluster)):
        div += np.power((New_Pi[k] - Pi[k]), 2)
        div += ((New_Mu[k]-Mu[k])*(New_Mu[k]-Mu[k]).transpose())[0,0]
        div += np.sum(np.multiply(New_Sigma[k]-Sigma[k],New_Sigma[k]-Sigma[k]))

    if cnt_repeat % 10 == 0:
        print(str(cnt_repeat) + " repetitions, div is " + str(div))

    Pi = copy.copy(New_Pi)
    for k in range(0, len(temp_cluster)):
        Mu[k] = New_Mu[k].copy()
        Sigma[k] = New_Sigma[k].copy()

    if div < EPISILON1:
        break

print("\n\n")
print("done!!!\n")

print("Results:")
print("Modelling using " + str(len(temp_cluster)) + " Gaussians")
for k in range(0, len(temp_cluster)):
    print("Pi_" + str(k) + ":    " + str(Pi[k]))
    print("Mu_" + str(k) + ":    " + str((Mu[k].flatten().tolist())[0]) )
    print("Sigma_" + str(k) + ":")
    print(Sigma[k])
    print(" ")

# visualization
if NUMBER_OF_FEATURE == 1:
    number_bins = 30
    ax1 = plt.subplot(221)
    ax1.hist((data.flatten().tolist())[0], density=True, histtype='stepfilled', alpha=0.2, bins = number_bins*len(temp_cluster))
    ax1.set_title('original plot')

    color_list = list(mcd.CSS4_COLORS)
    ax2 = plt.subplot(222)
    rv = [0] * len(temp_cluster)
    for k in range(0, len(temp_cluster)):
        rv[k] = norm(Mu[k][0,0], Sigma[k][0,0])
        ax2.hist((temp_cluster[k].flatten().tolist())[0], color = color_list[k], density=True, histtype='stepfilled', alpha=0.2, bins = number_bins)
    ax2.set_title('cluster')

    ax3 = plt.subplot(212)
    ax3.hist((data[:,0].flatten().tolist())[0], density=True, histtype='stepfilled', alpha=0.2, bins = number_bins*len(temp_cluster))
    x = np.linspace(0, 1, 500)
    y = np.linspace(0, 1, 500)
    for i in range(0,len(x)):
        y[i] = 0
        for k in range(0, len(temp_cluster)):
            y[i] += Pi[k] * rv[k].pdf(x[i]) / 10
    ax3.plot(x, y,'r-', lw=1, alpha=0.6, label='pdf')
    plt.savefig("fitting.pdf", bbox_inches='tight')
elif NUMBER_OF_FEATURE == 2:
    fig, ax = plt.subplots(1,2)
    ax[0].scatter((data[:,0].flatten().tolist())[0],(data[:,1].flatten().tolist())[0])
    ax[0].set(adjustable='box', aspect='equal')
    ax[0].set_title('original plot')

    color_list = list(mcd.CSS4_COLORS)
    for k in range(0, len(temp_cluster)):
        ax[1].scatter((temp_cluster[k][:,0].flatten().tolist())[0],
                      (temp_cluster[k][:,1].flatten().tolist())[0],
                      c = color_list[k])
    ax[1].set(adjustable='box', aspect='equal')
    ax[1].set_title('cluster plot')

    fig.savefig("clustering.pdf", bbox_inches='tight')
