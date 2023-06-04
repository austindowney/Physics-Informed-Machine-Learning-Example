#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 3: data processing

@author Austin Downey
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm

plt.close('all')

#%% Load and plot data
D = np.loadtxt('Vibration_measurement.txt',skiprows=23)

tt = D[:,0]
dd = D[:,1]

plt.figure('Beam Data',figsize=(6.5,3))
plt.plot(tt,dd,'-',label='data 1')
plt.grid(True)
plt.xlabel('time (s)')
plt.ylabel('acceleration (ms$^2$)')
plt.title('Beam Data')
plt.xlim([-0.1,45])
plt.tight_layout()
plt.savefig('beam data.png',dpi=300)



#%% Plot an FFT of the data

# Number of sample points
N = np.shape(dd)[0] # or dd.shape[0]
# sample spacing
T = (tt[-1]-tt[0])/tt.shape[0]
yf = sp.fftpack.fft(dd)
yyf = 2.0/N * np.abs(yf[0:N//2])
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.figure('FFt plot',figsize=(6.5,3))
plt.plot(xf,yyf)
plt.grid()
plt.xlim([0,150])
plt.xlabel('frequency (Hz)')
plt.ylabel('power')
plt.title('Beam FFT')
plt.tight_layout()
plt.savefig('FFT',dpi=300)

#%% Plot a spectrogram of the data

fs=1/T
x=dd 

plt.figure('Spectrogram',figsize=(6.5,3))
f, t, Sxx = sp.signal.spectrogram(x, fs,window=('tukey', 500), nperseg=10000, noverlap=5000)
plt.pcolormesh(t, f, Sxx,vmax=2)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.ylim([0,150])
plt.title('Beam Spectrogram')
plt.tight_layout()
plt.savefig('Spectrogram',dpi=300)


#%% Truncate the data for plotting

# limit the 3D plot to a set to Hz. 
f_limit = np.argmin(np.abs(f-150))
f_truncated = f[0:f_limit]

X, Y = np.meshgrid(t, f_truncated)
Z = Sxx[0:f_limit,:]



#%% interpolate the data onto a denser grid

rng = np.random.default_rng()
x = X.flatten()
y = Y.flatten()
z = Z.flatten()

X_dense = np.linspace(min(x), max(x),num=500)
Y_dense = np.linspace(min(y), max(y),num=500)
X_dense, Y_dense = np.meshgrid(X_dense, Y_dense)  # 2D grid for interpolation
interp = sp.interpolate.LinearNDInterpolator(list(zip(x, y)), z)
Z_dense = interp(X_dense, Y_dense)

plt.figure('Spectrogram Dense',figsize=(6.5,3))
plt.pcolormesh(X_dense, Y_dense, Z_dense,vmax=2)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.ylim([0,150])
plt.tight_layout()
plt.savefig('Spectrogram',dpi=300)



#%% Plot a spectrogram of the data in 3D


import math as math

def manual_log(data):
  if data < 10: # Linear scaling up to 1
    return data/10
  else: # Log scale above 1
    return math.log10(data)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(X_dense, Y_dense,  Z_dense, cmap=cm.viridis,rcount=200,ccount=200, antialiased=True,alpha=1,edgecolor='none', linewidth=0)#, linewidth=1)

ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_zlabel('acceleration (m/s$^2$)')
plt.title('Beam Spectrogram 3D')
plt.tight_layout()
ax.view_init(elev=54, azim=-61, roll=0)
# antialiased=False

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')
#ax.set_xlim([0,150])

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('Spectrogram_3D',dpi=300)



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
Z_dense_log = np.log(Z_dense)
surf = ax.plot_surface(X_dense, Y_dense,  Z_dense_log, cmap=cm.viridis,rcount=200,ccount=200, antialiased=True,alpha=1,edgecolor='none', linewidth=0)#, linewidth=1)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_zlabel('log$_{10}$[acceleration] (m/s$^2$)')
plt.title('Beam Spectrogram 3D in Semi-log')
ax.view_init(elev=54, azim=-61, roll=0)

# fig.colorbar(surf, shrink=0.5, aspect=15,location = 'left')
plt.tight_layout()

plt.savefig('Spectrogram_3D_Log',dpi=300)





