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

plt.close('all')

cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

#%% Define the input parameters

# considering a cantilever beam with nodal coordinates defined by the length of the beam
# and fixed at the left_hand side

beam_node_num = 100   # number of nodes, this must be at least 3
pin_location_start = 0.1 # The pin locaion in terms of L
pin_location_end = 0.5
consider_mass_drop = False # set whetherhe mass is dropped or not. 
calculate_temporal_response = True
mass_drop = 0.5 # set the mass drop as a portion of the number of steps
mass_location = 1.0 # The mass location in terms of L
mass_width = 0.025 # The 1-D width of the mass 
mass = 0.259 # the weight of the mass in kg
pin_node_rotation_spring = 900 # set the value of the spring at the pinned connection  
pin_node_start = int((beam_node_num*pin_location_start)-1)    # set the pin location in terms of matrix location
pin_node_end = int((beam_node_num*pin_location_end)-1)    # set the pin location in terms of matrix location
beam_length = 0.350     # length of the beam in meters
beam_width = 0.051   # width of the beamin meters
beam_height = 0.0063 # thickness of the beam in meters
beam_element = beam_node_num-1 # calculate the number of elements in the beam
beam_el_length = beam_length/beam_element     # calculate the element lengths of the beam
mass_el_number = int(mass_width/beam_el_length) # defines the number of elements with added mass
# define the mass of the elements as a function of the dropped mass
if mass_width/beam_el_length <= 2:
    mass_el = mass
else:
    mass_el = mass/mass_el_number 
# define what nodes these masses are attached to
mass_node = np.zeros((beam_node_num),dtype=bool)
ss1 = int(int(beam_node_num*mass_location)-(mass_el_number-1)/2)
ss2 = ss1+mass_el_number
if ss2 > beam_node_num:
    ss2 = beam_node_num
    ss1 = beam_node_num - mass_el_number
mass_node[ss1:ss2+1]=True
beam_E = 200*1e9    # Youngs modules of steel in Pa
beam_density = 7800 # density of steel in kg/m^3
beam_I = (beam_width*beam_height**3)/12 # caclulated moment of inertia
beam_area = beam_width*beam_height
dt = 0.0001 # Time step.
steps =np.abs(pin_node_start-pin_node_end) # Number of steps for loop below.
print('This will run for ' + str(steps-1) + ' steps')
C_alpha = 0
C_beta = 0.0003 # was 0.00003
gravity = -9.807 # gravity in m/s^2
pin_nodes = np.linspace(pin_node_start,pin_node_end,steps,dtype=int) # set the array of pin locations
pin_locations = pin_nodes/beam_node_num
tt = np.arange(0,dt*steps,dt) # Time vector.

#%% Solve the simple beam case analytically
analytical_tip_load = (mass*gravity*beam_length**3)/(3*beam_E*beam_I)
analytical_distubted_load = (beam_density*beam_area*9.81*beam_length**4)/(8*beam_E*beam_I)
analytical = analytical_tip_load+analytical_distubted_load

#%% build all the main matrices, M, K,.

# build a list of displacement and rotations
DOF = []
for i in range(beam_node_num):
    DOF.append('displacement')
    DOF.append('rotation')
DOF = np.asarray(DOF)
    
# define the mass matrix of a Euler-Bernoulli beam
M_el = (beam_density*beam_area*beam_el_length)/420* \
np.matrix([[156,22*beam_el_length,54,-13*beam_el_length], \
           [22*beam_el_length,4*beam_el_length**2,13*beam_el_length,-3*beam_el_length**2], \
           [54,13*beam_el_length,156,-22*beam_el_length], \
           [-13*beam_el_length,-3*beam_el_length**2,-22*beam_el_length,4*beam_el_length**2]])

M_mass_el = mass_el/420* \
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
M_no_mass = np.zeros((matrix_size,matrix_size))
M_mass = np.zeros((matrix_size,matrix_size))
F_static_mass = np.zeros((matrix_size))
K_global = np.zeros((matrix_size,matrix_size))

# for each element, add the element matrix into the global matirx
for elem_num in range(0,beam_element):
    n = (elem_num)*2
    M_no_mass[n:n+4,n:n+4] = np.add(M_no_mass[n:n+4,n:n+4],M_el)
    K_global[n:n+4,n:n+4] = np.add(K_global[n:n+4,n:n+4],K_el)

for elem_num in range(0,beam_element):
    if mass_node[elem_num] == True:
        n = (elem_num)*2
        M_mass[n:n+4,n:n+4] = np.add(M_mass[n:n+4,n:n+4],M_mass_el)

  
#%% Static solution

# Build the F matix used to solve the static solution
F_static = np.zeros((matrix_size))
F_static[DOF=='displacement'] = 1
F_static[0] = 0.5
F_static[-2] = 0.5
F_static = F_static*beam_density*beam_area*beam_el_length*gravity

# add the drop mass to the F static vector
for elem_num in range(0,beam_element):
    if mass_node[elem_num] == True:
        n = (elem_num)*2
        F_static_mass[n:n+4] = np.add(F_static_mass[n:n+4],np.matrix([1,0,1,0])*0.5)
F_static_mass = F_static_mass*mass_el*gravity

# rebuild the F_static vector
F_static = np.add(F_static,F_static_mass)

# remove the row and column associated with the pin location. 
K_static  = np.delete(np.delete(K_global,(pin_node_start*2),axis=0),(pin_node_start*2),axis=1)
F_static = np.delete(F_static,(pin_node_start*2),axis=0)
DOF_static  = np.delete(DOF,(pin_node_start*2),axis=0)

# for the fixed end on the left side, u_1 and u_2 = 0, so we can remove these columns
# and rows form the matrixes. 
# apply the boundary conditions
K_static  = np.delete(np.delete(K_static,(0,1),axis=0),(0,1),axis=1)
F_static = np.delete(F_static,(0,1),axis=0)
DOF_static  = np.delete(DOF_static,(0,1),axis=0)

DOF_displacement_static = DOF_static =='displacement'

X_static = np.matmul(F_static,np.linalg.inv(K_static))
#X_static = np.matmul(np.linalg.inv(K),F_static)
X_static_displacement = X_static[0:DOF_displacement_static.shape[0]][DOF_displacement_static]

# Build a list of the non zero displacements and map this onto the empty zeros matrix
displacements_static = np.zeros((beam_node_num))
displacements_non_zero = np.ones((beam_node_num),dtype=bool)
displacements_non_zero[0] = False
displacements_non_zero[pin_node_start] = False
displacements_static[displacements_non_zero] = X_static_displacement

plt.figure()
plt.plot(displacements_static*1000)
plt.grid(True)
plt.xlabel('nodes (#)')
plt.ylabel('displacement (mm)')
plt.title('Static displacement over the length of the beam')
plt.savefig('figures/discrete_static_solution_displacement_'+str(beam_node_num),dpi=300)
plt.legend()
plt.tight_layout()


#%% State Space discrete solution

X = np.zeros((K_static.shape[0]*2,steps))
X[0:X_static.shape[0],0] = np.copy(X_static)# set the initial X to the static displacement values
displacements = np.zeros((beam_node_num,steps)) 
frequencies = np.zeros((X_static.shape[0],steps)) 
frequencies[frequencies==0]='NaN'
# Build a list of the non zero displacements and map this onto the empty zeros matrix
displacements_discrete = np.zeros((beam_node_num))
displacements_non_zero = np.ones((beam_node_num),dtype=bool)
displacements_non_zero[0] = False
displacements_non_zero[pin_nodes[0]] = False

# Build a list of the non zero displacements and map this onto the empty zeros matrix
displacements[displacements_non_zero,0] = X_static_displacement

for i_step in np.arange(steps-1):
    tt_in=time.time()
    # set the pin node
    pin_node = pin_nodes[i_step]

    #set the mass matrix
    if consider_mass_drop == True:
        if i_step <= steps*mass_drop:
            M = np.add(M_mass,M_no_mass) # the mass matrix has the added mass
        else:
            M=M_no_mass
    else:
        M=M_no_mass

    K = np.copy(K_global)

    # add the stffness from the pin roller
    node_rotation_cell = pin_node*2+1
    K[node_rotation_cell,node_rotation_cell] = K[node_rotation_cell,node_rotation_cell] +pin_node_rotation_spring


    # remove the row and column associated with the pin location. 
    M = np.delete(np.delete(M,(pin_node*2),axis=0),(pin_node*2),axis=1)
    K = np.delete(np.delete(K,(pin_node*2),axis=0),(pin_node*2),axis=1)
    DOF_step = np.delete(DOF,(pin_node*2),axis=0)

    # for the fixed end on the left side, u_1 and u_2 = 0, so we can remove these columns
    # and rows form the matrixes. 
    # apply the boundary conditions
    M = np.delete(np.delete(M,(0,1),axis=0),(0,1),axis=1)
    K = np.delete(np.delete(K,(0,1),axis=0),(0,1),axis=1)
    DOF_step = np.delete(DOF_step,(0,1),axis=0)


    # Calculation of the natural frequencies. 
    eigvals,eigvects = sp.linalg.eig(K,M)
    eigvals=np.expand_dims(np.real(eigvals), axis=0)
    frequencies[:,i_step+1] = np.sort(np.real(np.squeeze(np.sqrt(eigvals))))/(2*np.pi) # Natural frequencies, in Hz

    if calculate_temporal_response == True:
        # define the damping coefficent
        C = C_alpha*M+C_beta*K
        
        zeros_matrix = np.zeros((M.shape[0],M.shape[0]))
        identity_matrix = np.eye(M.shape[0])
        
        A00 = np.zeros((M.shape[0],M.shape[0]))
        A01 = np.eye(M.shape[0])
        A10 = np.matmul(np.linalg.inv(M),-K)
        A11 = np.matmul(np.linalg.inv(M),-C)
        A = np.vstack((np.hstack((A00,A01)),np.hstack((A10,A11))))
        
        B_p = np.vstack((zeros_matrix,np.linalg.inv(M))) # define the B matrix that considers the applied force
        B_g = np.vstack((zeros_matrix,identity_matrix)) # define the B matrix that considers gravity
    
        P_p = np.zeros((B_p.shape[1],steps))# Initialize the force vector. 
        P_g = np.ones((B_g.shape[1]))*gravity# Initialize the force vector. 
        P_p[-1,0]=-100 # define the location of the input loading. -1 is the tip
        
        DOF_displacement = DOF_step=='displacement'
        
        
        #%% discrete solution
        
        # Solve the state space equations. 
        # solve these out-side the loop to increase speed
        aa = sp.linalg.expm(A*dt)
        aaa = np.matmul(np.linalg.inv(A),np.subtract(aa,np.eye(A.shape[0])))
    
        b_p = np.matmul(B_p,P_p[:,i_step])
        b_g = np.matmul(B_g,P_g)
        bbb = np.add(b_p,b_g)
        X[:,i_step+1] = np.matmul(aa,X[:,i_step]) + np.matmul(aaa,bbb)
    
        X_displacement = X[0:DOF_displacement.shape[0],i_step+1][DOF_displacement] 
        
        # Build a list of the non zero displacements and map this onto the empty zeros matrix
        displacements_discrete = np.zeros((beam_node_num))
        displacements_non_zero = np.ones((beam_node_num),dtype=bool)
        displacements_non_zero[0] = False
        displacements_non_zero[pin_node] = False
    
        # Build a list of the non zero displacements and map this onto the empty zeros matrix
        displacements[displacements_non_zero,i_step+1] = X_displacement
    
    tt_out=time.time()
    print(str(i_step) +'; '+ str(np.round(frequencies[0:5,i_step]))+'; run time = '+str(np.round((tt_out-tt_in),3))+' s')
    

data = {'tt':tt,
        'frequencies':frequencies,
        'displacements':displacements,
        'pin_nodes':pin_nodes,
        'pin_locations':pin_locations,
        'beam_node_num':beam_node_num,
        'displacements_static':displacements_static,
        'mass_drop': mass_drop,
        'mass_location': mass_location,
        'mass_width': mass_width,
        'mass': mass,
        'pin_node_rotation_spring': pin_node_rotation_spring,
        'pin_node_start': pin_node_start,
        'pin_node_end': pin_node_end,
        'beam_length': beam_length,
        'beam_width': beam_width,
        'beam_height': beam_height,
        'consider_mass_drop':False,
        'calculate_temporal_response':True  
        }

pickle.dump(data,open('data/'+str(beam_node_num)+'_nodes.pickle','wb'))

# Plot the displacement at the tip
plt.figure()
for i in range(steps):
    plt.plot(displacements[:,i]*1000,label='time step '+str(i))
plt.grid(True)
plt.xlabel('nodes (#)')
plt.ylabel('displacement (mm)')
plt.title('dynamic displacement over the length of the beam  for ' + str(beam_node_num) + ' elements')
plt.savefig('figures/_simulation_displacement_'+str(beam_node_num),dpi=300)
plt.legend()
plt.tight_layout()

# polt selected beam displacements    
#displacement_times = [0,100,250,500,750,9999]
#plt.figure()
#for i in displacement_times:
#    plt.plot(displacements[:,i]*1000,label='time step '+str(i))
#plt.grid(True)
#plt.xlabel('nodes (#)')
#plt.ylabel('displacement (mm)')
#plt.title('displacement over the length of the beam')
#plt.savefig('figures/discrete_solution_displacement',dpi=300)
#plt.legend()
#plt.tight_layout()

# plot the final displaced shape
plt.figure()
plt.plot(tt,displacements[-1,:]*1000,'.-')
plt.grid(True)
plt.xlabel('time (s)')
plt.ylabel('displacement (mm)')
plt.title('displacement at last node for ' + str(beam_node_num) + ' elements')
plt.savefig('figures/_simulation_discrete_solution_tip_'+str(beam_node_num),dpi=300)
plt.tight_layout()
#plt.xlim([0,3.7])
#plt.savefig('figures/discrete_solution',dpi=300)


# Plot the displacement at the tip
plt.figure()
for i in range(3):
    plt.plot(pin_locations,frequencies[i,:],label='mode frequency '+str(i+1))
plt.grid(True)
plt.xlabel('pin locations (% beam)')
plt.ylabel('frequency (Hz)')
plt.title('Frequency of first 3 modes for ' + str(beam_node_num) + ' elements')
plt.savefig('figures/_simulation_frequency_vs_time_'+str(beam_node_num),dpi=300)
plt.legend()
plt.tight_layout()







