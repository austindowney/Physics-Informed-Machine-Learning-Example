# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% import modules
import IPython as IP
IP.get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

#%% Load the data

data_20 = np.load('data/20_nodes.pickle',allow_pickle=True)
data_25 = np.load('data/25_nodes.pickle',allow_pickle=True)
data_30 = np.load('data/30_nodes.pickle',allow_pickle=True)
data_35 = np.load('data/35_nodes.pickle',allow_pickle=True)
data_40 = np.load('data/40_nodes.pickle',allow_pickle=True)
data_45 = np.load('data/45_nodes.pickle',allow_pickle=True)
data_50 = np.load('data/50_nodes.pickle',allow_pickle=True)
data_75 = np.load('data/75_nodes.pickle',allow_pickle=True)
data_100 = np.load('data/100_nodes.pickle',allow_pickle=True)
data_200 = np.load('data/200_nodes.pickle',allow_pickle=True)
data_400 = np.load('data/400_nodes.pickle',allow_pickle=True)
data_600 = np.load('data/600_nodes.pickle',allow_pickle=True)
data_800 = np.load('data/800_nodes.pickle',allow_pickle=True)


#%% Plot the frequency responses for a given pin location

plt.figure(figsize=(6,4))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.grid(True)
    plt.plot(data_20['pin_locations']*100,data_20['frequencies'][i,:],'-',color=cc[0],label=str(20)+' nodes')
    plt.plot(data_50['pin_locations']*100,data_50['frequencies'][i,:],'--',color=cc[1],label=str(50)+' nodes')
    plt.plot(data_100['pin_locations']*100,data_100['frequencies'][i,:],'-.',color=cc[2],label=str(100)+' nodes')
    plt.plot(data_800['pin_locations']*100,data_800['frequencies'][i,:],':',color=cc[3],label=str(800)+' nodes')
    plt.xlim([9,51])
    #plt.ylim([100,10000])

plt.subplot(2,2,1)
plt.ylabel('frequency (Hz)')
plt.title('first frequency')
plt.subplot(2,2,2)
plt.title('second frequency')
plt.legend(framealpha=1,fontsize=8)
plt.subplot(2,2,3)
plt.title('third frequency')
plt.ylabel('frequency (Hz)')
plt.xlabel('roller locations (% beam)')
plt.subplot(2,2,4)
plt.title('forth frequency')
plt.xlabel('roller locations (% beam)')
plt.tight_layout()
plt.savefig('simulation_1.png',dpi=300)


#%% Plot the convergence test

error = np.zeros((4,13))
for i in range(4):

    x = data_800['pin_locations']
    y = data_800['frequencies'][i,:]

    yinterp = np.interp(x, data_20['pin_locations'],data_20['frequencies'][i,:])
    error[i,0] = np.nanmean(np.abs(y-yinterp))

    yinterp = np.interp(x, data_25['pin_locations'],data_25['frequencies'][i,:])
    error[i,1] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_30['pin_locations'],data_30['frequencies'][i,:])
    error[i,2] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_35['pin_locations'],data_35['frequencies'][i,:])
    error[i,3] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_40['pin_locations'],data_40['frequencies'][i,:])
    error[i,4] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_45['pin_locations'],data_45['frequencies'][i,:])
    error[i,5] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_50['pin_locations'],data_50['frequencies'][i,:])
    error[i,6] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_75['pin_locations'],data_75['frequencies'][i,:])
    error[i,7] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_100['pin_locations'],data_100['frequencies'][i,:])
    error[i,8] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_200['pin_locations'],data_200['frequencies'][i,:])
    error[i,9] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_400['pin_locations'],data_400['frequencies'][i,:])
    error[i,10] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_600['pin_locations'],data_600['frequencies'][i,:])
    error[i,11] = np.nanmean(np.abs(y-yinterp))
    
    yinterp = np.interp(x, data_800['pin_locations'],data_800['frequencies'][i,:])
    error[i,12] = np.nanmean(np.abs(y-yinterp))


# plot the convergence test   
plt.figure(figsize=(6,4))
xx = [20,25,30,35,40,45,50,75,100,200,400,600,800]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.grid(True)
    plt.plot(xx,error[i,:])

plt.subplot(2,2,1)
plt.ylabel('mean absolute error (Hz)')
plt.title('first frequency')
plt.ylim(-1,11)
plt.subplot(2,2,2)
plt.title('second frequency')
plt.ylim(-5,65)
plt.subplot(2,2,3)
plt.title('third frequency')
plt.ylabel('mean absolute error (Hz)')
plt.xlabel('nodes in FEA model')
plt.ylim(-10,151)
plt.subplot(2,2,4)
plt.title('forth frequency')
plt.xlabel('nodes in FEA model')
plt.ylim(-15,201)
plt.tight_layout()

plt.savefig('simulation_2.png',dpi=300)
















