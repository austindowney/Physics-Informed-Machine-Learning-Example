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

import numpy as np
import matplotlib.tri as mtri
import plotly.graph_objects as go

plt.close('all')

#%% Load and plot data
D = np.loadtxt('Vibration_measurement.txt',skiprows=23)

tt = D[:,0]
dd = D[:,1]


#%% spectrogram of the data


T = (tt[-1]-tt[0])/tt.shape[0]
fs=1/T
x=dd 

f, t, Sxx = sp.signal.spectrogram(x, fs,window=('tukey', 500), nperseg=10000, noverlap=5000)


#%% Truncate the data for plotting

# limit the 3D plot to a set to Hz. 
f_limit = np.argmin(np.abs(f-150))
f_truncated = f[0:f_limit]

X, Y = np.meshgrid(t, f_truncated)
Z = Sxx[0:f_limit,:]



#%% interpolate the data onto a denser grid


x = X.flatten()
y = Y.flatten()
z = Z.flatten()

X_dense = np.linspace(min(x), max(x),num=200)
Y_dense = np.linspace(min(y), max(y),num=200)
X_dense, Y_dense = np.meshgrid(X_dense, Y_dense)  # 2D grid for interpolation
interp = sp.interpolate.LinearNDInterpolator(list(zip(x, y)), z)
Z_dense = interp(X_dense, Y_dense)


#%% Plot a spectrogram of the data in 3D



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
Z_dense_log = np.log(Z_dense)
Z_log = np.log(Z)
surf = ax.plot_surface(X, Y,  Z_log, cmap=cm.viridis,rcount=200,ccount=200, antialiased=True,alpha=1,edgecolor='none', linewidth=0)#, linewidth=1)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_zlabel('log$_{10}$[acceleration] (m/s$^2$)')
plt.title('Beam Spectrogram 3D Log Scape')
ax.view_init(elev=54, azim=-61, roll=0)

# fig.colorbar(surf, shrink=0.5, aspect=15,location = 'left')
plt.tight_layout()

# plt.savefig('Spectrogram_3D_Log',dpi=300)


#%% Plot a spectrogram of the data in interactive HTML plot



# Map radius, angle pairs to x, y, z points.
x = X_dense.flatten()
y = Y_dense.flatten()
z = Z_dense_log.flatten()

### TRIANGULATION
# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = mtri.Triangulation(x, y)


triangles = triang.triangles

### PLOT
fig = go.Figure(data=[
    # go.Mesh allows to provide the triangulation
    go.Mesh3d(
        x=x, y=y, z=z,
        colorbar_title='log<sub>10<\sub>[acceleration] (m/s<sup>2</sup>)',
        colorscale="viridis",
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity =z,
        # i, j and k give the vertices of triangles
        i = triangles[:, 0],
        j = triangles[:, 1],
        k = triangles[:, 2],
        showscale=True
    )
])

fig.update_layout(
    title='Beam Data', 
    autosize=True,
    width=1200, 
    height=900,
    #margin=dict(l=65, r=50, b=65, t=90),
    scene=dict(
        xaxis_title='time (s)',
        yaxis_title='frequency (hz)',
        zaxis_title='log<sub>10</sub>[acceleration] (m/s<sup>2</sup>)',
    ),
)

fig.show()

### EXPORT TO HTML
# Please, execute `help(fig.write_html)` to learn about all the
# available keyword arguments to control the output
fig.write_html("3D_plot.html", include_plotlyjs=True, full_html=True)





























