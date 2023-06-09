# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% import modules
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from matplotlib import cm
import time
import subprocess
import pickle
import scipy.io as sio
import sympy as sym
from matplotlib import cm
import re as re
from scipy import signal

plt.close('all')

cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

#%% Load the data

def frequency_modes(mass_on=True, beam_node_num = 100, pin_node =20, number_modes=5):
    # considering a cantilever beam with nodal coordinates defined by the length of the beam
    # and fixed at the left-hand side
    
    #beam_node_num = 100   # number of nodes, this must be at least 3
    #pin_location = 0.5 #25 # The pin locaion in terms of L
    pin_node_rotation_spring = 0 # set the value of the spring at the pinned connection  
    #pin_node = int((beam_node_num*pin_location)-1)    # set the pin location in terms of matrix location
    beam_length = 0.350     # length of the beam in meters
    beam_width = 0.051   # width of the beamin meters
    beam_height = 0.0063 # thickness of the beam in meters
    beam_element = beam_node_num-1 # calculate the number of elements in the beam
    beam_el_length = beam_length/beam_element     # calculate the element lengths of the beam
    beam_E = 200*1e9    # Youngs modules of steel in Pa
    beam_density = 7800 # density of steel in kg/m^3
    beam_I = (beam_width*beam_height**3)/12 # caclulated moment of inertia
    beam_area = beam_width*beam_height
    dt = 0.0005 # Time step.
    steps = 10000 # Number of steps for loop below.
    
    # define the mass matrix of a Euler-Bernoulli beam
    M_el = (beam_density*beam_area*beam_el_length)/420* \
    np.matrix([[156,22*beam_el_length,54,-13*beam_el_length], \
               [22*beam_el_length,4*beam_el_length**2,13*beam_el_length,-3*beam_el_length**2], \
               [54,13*beam_el_length,156,-22*beam_el_length], \
               [-13*beam_el_length,-3*beam_el_length**2,-22*beam_el_length,4*beam_el_length**2]])
        
    # define the stiffness matrix of a Euler-Bernoulli beam
    K_el = (beam_E*beam_I)/beam_el_length**3* \
    np.matrix([[12,6*beam_el_length,-12,6*beam_el_length], \
               [6*beam_el_length,4*beam_el_length**2,-6*beam_el_length,2*beam_el_length**2], \
               [-12,-6*beam_el_length,12,-6*beam_el_length], \
               [6*beam_el_length,2*beam_el_length**2,-6*beam_el_length,4*beam_el_length**2]])
    
    matrix_size = (beam_node_num)*2
    M = np.zeros((matrix_size,matrix_size))
    K = np.zeros((matrix_size,matrix_size))
    
    # for each element, add the element matrix into the global matirx
    for elem_num in range(0,beam_element):
        n = (elem_num)*2
        M[n:n+4,n:n+4] = np.add(M[n:n+4,n:n+4],M_el)
        K[n:n+4,n:n+4] = np.add(K[n:n+4,n:n+4],K_el)

    # add the stffness from the pin roller
    node_rotation_cell = pin_node*2+1
    K[node_rotation_cell,node_rotation_cell] = K[node_rotation_cell,node_rotation_cell] +pin_node_rotation_spring
    
    # remove the row and column associated with the pin location. 
    M = np.delete(np.delete(M,(pin_node*2),axis=0),(pin_node*2),axis=1)
    K = np.delete(np.delete(K,(pin_node*2),axis=0),(pin_node*2),axis=1)

    
    # for the fixed end on the left side, u_1 and u_2 = 0, so we can remove these columns
    # and rows form the matrixes. 
    # apply the boundary conditions
    M = np.delete(np.delete(M,(0,1),axis=0),(0,1),axis=1)
    K = np.delete(np.delete(K,(0,1),axis=0),(0,1),axis=1)
   
    # Calculation of the natural frequencies. 
    eigvals,eigvects = sp.linalg.eig(K,M)
    eigvals=np.expand_dims(np.real(eigvals), axis=0)
    FEA_wn = np.sort(np.real(np.squeeze(np.sqrt(eigvals)))) # Natural frequencies, rad/s
    Frequencies = FEA_wn/(2*np.pi) # Natural freq in Hz
    return(Frequencies[0:number_modes])










