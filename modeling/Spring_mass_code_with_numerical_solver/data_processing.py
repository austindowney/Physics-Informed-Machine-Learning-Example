# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%% import modules
import IPython as IP
IP.get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import os as os
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from matplotlib import cm
import time
import subprocess
import pickle
import scipy.io as sio
import sympy as sym
from matplotlib import cm
import re as re
from scipy import signal
from scipy import fft
import json as json
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d as mp3d
from scipy.integrate import odeint


# set default fonts and plot colors
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'}) # I don't think I need this as its set to 'stixsans' above.

cc = plt.rcParams['axes.prop_cycle'].by_key()['color']


plt.close('all')



#%% Define functions

def equation_of_motion(y, t, K, M, F, omega):
    """
    Represents the equation of motion for a mass-spring system with two degrees of freedom using matrix notation.
    
    Parameters:
        y: array-like
            Current values of the positions and velocities.
        t: float
            Current time.
        K: array-like
            Stiffness matrix.
        M: array-like
            Mass matrix.
    
    Returns:
        array-like
            The derivatives of the positions and velocities.
    """
    DOF = k.shape[0]
    x = y[:DOF]  # Positions
    v = y[DOF:]  # Velocities
    dxdt = v
    #dvdt = np.linalg.solve(M, -np.dot(K, x))
    dvdt = np.linalg.solve(M, -np.dot(K, x) + F * np.sin(omega * t))
    return np.concatenate((dxdt, dvdt))


# Function to fill offset diagonal of the matrix
def fill_offset_diagonal(matrix, offset, values):
    """
    Fills the offset diagonal of the matrix with the given values.
    """
    np.fill_diagonal(matrix[offset:], values[:matrix.shape[0]-offset])
    np.fill_diagonal(matrix[:, offset:], values[:matrix.shape[0]-offset])
    return matrix

#%% Process the data

# Define the degree of freedom
DOF = 10

# Initialize the stiffness matrix
k = np.zeros([DOF, DOF])

# Fill the diagonals of the stiffness matrix
fill_offset_diagonal(k, 0, np.ones(DOF) * 2)
fill_offset_diagonal(k, 1, np.ones(DOF) * -1)

# Define the mass matrix
m = np.identity(DOF)

# Calculate eigenvalues and eigenvectors for the mode shapes and natural frequencies
eig_value, eig_vect = sp.linalg.eig(k, m)

# Extract the natural frequencies
omega_1 = np.round(np.real(np.sqrt(eig_value[0])), 5)
omega_2 = np.round(np.real(np.sqrt(eig_value[1])), 5)
omega_3 = np.round(np.real(np.sqrt(eig_value[2])), 5)

# Extract the eigenvectors
v_1 = eig_vect[:, 0]
v_2 = eig_vect[:, 1]
v_3 = eig_vect[:, 2]

# Set up external force
F = np.zeros(DOF)  # Amplitude of the external force for each degree of freedom
F[4] = 0.1
omega = omega_3  # Angular frequency of the forcing function

# Initial conditions
x0 = np.zeros(DOF)  # Initial positions
x0[4] = 0.0
v0 = np.zeros(DOF)  # Initial velocities
y0 = np.concatenate((x0, v0))  # Initial state

# Time points
t = np.linspace(0, 60, 500)  # 100 points from 0 to 10 seconds

# Solve the ODE
def equation_of_motion(y, t, k, m, F, omega):
    """
    Defines the equation of motion for the system.
    """
    x = y[:DOF]
    v = y[DOF:]
    dxdt = v
    dvdt = np.linalg.solve(m, F * np.sin(omega * t) - np.dot(k, x))
    return np.concatenate((dxdt, dvdt))

solution = odeint(equation_of_motion, y0, t, args=(k, m, F, omega))

# Extract positions and velocities from the solution
solution_x = solution[:, :DOF]
solution_v = solution[:, DOF:]

# Plotting
fig = plt.figure(figsize=(6.5, 2.75))
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(solution_x.T, aspect=2, origin='lower', extent=[0, np.max(t), 0, DOF])
ax.set_xlabel('time (s)')
ax.set_ylabel('position (DOF)')
cbar = plt.colorbar(im)
cbar.set_label("displacement (m)")
plt.savefig('omega_3.jpg',dpi=250)





