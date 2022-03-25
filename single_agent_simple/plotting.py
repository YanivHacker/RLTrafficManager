import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import figure
from pylab import rcParams
import glob
import os
from pathlib import Path
import configparser
import binascii

params = { 
    'axes.labelsize': 13, 
    'font.size': 13, 
    'legend.fontsize': 12, 
    'xtick.labelsize': 15, 
    'ytick.labelsize': 13, 
    'figure.figsize': [7.8, 4.1]
}
FS=(4.9, 4.4)
rcParams.update(params)


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'

def plot_data2(data_matrix, ax, lcolor, lline, label, p, lmarker=''):
    plot_only_each_n_point=100
    #import ipdb; ipdb.set_trace()
    data_matrix=data_matrix.reshape((int(data_matrix.shape[0]*plot_only_each_n_point),int(data_matrix.shape[1]/plot_only_each_n_point)), order='F')
    x = np.arange(0, data_matrix.shape[1])
    x2= np.arange(0, data_matrix.shape[1])*p*plot_only_each_n_point
    lx = np.arange(0, data_matrix.shape[1], 1)
    lx2 = np.arange(0, data_matrix.shape[1])*p*plot_only_each_n_point
    ymean = np.mean(data_matrix, axis=0)
    ys=data_matrix
    ystd = np.std(ys, axis=0)
    ystderr = ystd / np.sqrt(len(ys))
    ax.fill_between(lx2*p, ymean - ystderr, ymean + ystderr, color=lcolor, alpha=.4)
    ax.fill_between(lx2*p, ymean - ystd,    ymean + ystd,    color=lcolor, alpha=.2)
    ax.plot(lx2*p, ymean[lx], linewidth=2, linestyle=lline, color=lcolor, label=label,
            marker=lmarker, markevery=1000, markersize=4.5)
    
  
def plot_from_file(path, fig, ax, color, label, sign):
    results2 = []
    print("path", path)
    for directory in glob.iglob(path+'**/output/', recursive=True):
        print(directory)
        results = []
        for index in range(1,4):
            for filename in glob.iglob(directory+'**_run'+str(index)+'.csv', recursive=False):
                if is_gz_file(filename):
                    subresults = pd.read_csv(filename, compression='gzip', sep=",", dtype=np.float)
                else:
                    subresults = pd.read_csv(filename, sep=",", dtype=np.float)
                subresults = subresults['total_wait_time'][0:19900]
                # import ipdb; ipdb.set_trace()
                 
                results.append(subresults)

        results = np.array(results)
        results = results.reshape(results.shape[0]*results.shape[1])
        results2.append(results)
    results2 = np.array(results2)
    plot_data2(results2, ax, color, sign, label, 1)
    return ax

def plot_from_file_paraworker(path, fig, ax, color, lable, sign):
    nbworker=8
    results2 = []
    print("path", path)
    for directory in glob.iglob(path+'**/output/', recursive=True):
        print(directory)
        results = []
        for index in range(1,4):
            subresults2=[]
            for worker in range(0,nbworker):
                for filename in glob.iglob(directory+'**'+str(worker)+'_conn0_run'+str(index)+'.csv', recursive=False):
                    if is_gz_file(filename):
                        subresults = pd.read_csv(filename, compression='gzip', sep=",", dtype=np.float)
                    else:
                        subresults = pd.read_csv(filename, sep=",", dtype=np.float)
                    subresults = subresults['total_wait_time'][0:19900]
                    subresults2.append(subresults)
            subresults2 = np.array(subresults2).mean(axis=0)
            results.append(subresults2)
        results = np.array(results)
        results = results.reshape(results.shape[0]*results.shape[1])
        results2.append(results)
    
    results2 = np.array(results2)
    plot_data2(results2, ax, color, sign, lable, 1)
    return ax

path_local="/home/umer/exp/sumo/"
drawlist=['ppo', 'a2c', 'fixed', 'random', 'all', 'dqn']    
# drawlist = ["fixedrandom", "all"]

my_labels= {"x1": "A2C", "x2": "PPO", "x3": "DQN", "x4": "Random", "x5": "Fixed"}

if "fixedrandom" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file(path_local + "sumo_2way_single/fixed_"+'**', fig, ax, 'lime', my_labels["x5"], ':')
    plot_from_file(path_local + "sumo_2way_single/randomm_"+'**', fig, ax, 'grey', my_labels["x4"], ':')
    
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Waiting Time")
    ax.legend(loc="best", ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.savefig('random_vs_fix_sumo_curves.png')

if "dqn" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file_paraworker(path_local + "sumo_2way_single/dqn_"+'**', fig, ax, 'magenta', my_labels["x3"], '-')
    
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Waiting Time")
    ax.legend(loc="best", ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.savefig('dqn_sumo_curves.png')

if "ppo" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file_paraworker(path_local + "sumo_2way_single/ppo_"+'**', fig, ax, 'brown', my_labels["x2"], '--')
    
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Waiting Time")
    ax.legend(loc="best", ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.savefig('ppo_sumo_curves.png')

if "a2c" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file_paraworker(path_local + "sumo_2way_single/a2c_"+'**', fig, ax, 'orange', my_labels["x1"], '--')
    
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Waiting Time")
    ax.legend(loc="best", ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.savefig('a2c_sumo_curves.png')

if "random" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file(path_local + "sumo_2way_single/randomm_"+'**', fig, ax, 'grey', my_labels["x4"], ':')
    
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Waiting Time")
    ax.legend(loc="best", ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.savefig('random_sumo_curves.png')

if "fixed" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file(path_local + "sumo_2way_single/fixed_"+'**', fig, ax, 'lime', my_labels["x5"], ':')
    
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Waiting Time")
    ax.legend(loc="best", ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.savefig('fixed_sumo_curves.png')

if "all" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file_paraworker(path_local + "sumo_2way_single/ppo_"+'**', fig, ax, 'brown', my_labels["x2"], '--')
    plot_from_file_paraworker(path_local + "sumo_2way_single/a2c_"+'**', fig, ax, 'orange', my_labels["x1"], '-')
    plot_from_file_paraworker(path_local + "sumo_2way_single/dqn_"+'**', fig, ax, 'magenta', my_labels["x3"], '-')
    plot_from_file(path_local + "sumo_2way_single/fixed_"+'**', fig, ax, 'lime', my_labels["x5"], ':')
    plot_from_file(path_local + "sumo_2way_single/randomm_"+'**', fig, ax, 'grey', my_labels["x4"], ':')
    
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Rewards")
    ax.set_title('Learning Curves')
    ax.legend(loc="best", ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig('learning_curves_all.png')