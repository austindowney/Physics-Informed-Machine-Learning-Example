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
import FEA_function as FEA


plt.close('all')

cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 


#%% Set the code parameters

# set the tunning parameters
beam_node_num_FEA = 10 # set the number of elements used in the FEA model that will track the beam
particle_number = 3 # the number of particles that will be tested each time 
noise_level = 0 # the level of noise in the system. 
step_pin_node = beam_node_num_FEA*0.6 # set the initial positon of the pin
step_pin_standard_dev = 0.5 #1/3*particle_number # the standard dev of the pin PDF in terms of pins

#%% Load the data

# load the data from the test
data_exp_input = np.load('data/800_nodes.pickle',allow_pickle=True)
frequencies_exp_input = data_exp_input['frequencies'][:,1::]
pin_locations_exp_input = data_exp_input['pin_locations'][1::]
pin_nodes_exp_input = data_exp_input['pin_nodes'][1::]
beam_node_num_exp = data_exp_input['beam_node_num']

# define the input into the system and apply noise to the "experimental data"
frequencies_exp = frequencies_exp_input+np.random.randn(frequencies_exp_input.shape[0],frequencies_exp_input.shape[1])*noise_level
pin_locations_exp = pin_locations_exp_input
pin_nodes_exp = pin_nodes_exp_input

# calculate the locations of the pins in the system
pin_locations_FEA = np.linspace(0,0.35,beam_node_num_FEA,endpoint=True)/0.35

# get the number of data points to solve for
data_length = frequencies_exp.shape[1]


#%% plot the input data
plt.figure(figsize=(6.5,5))
plt.subplot(211)
plt.plot(pin_locations_exp)
plt.title('pin location')
plt.grid('on')
plt.xlabel('step number (#)')
plt.ylabel('pin location (% beam)')
plt.legend()

plt.subplot(212)
plt.plot(pin_locations_exp,frequencies_exp[0,:]-frequencies_exp[0,0],':',label='fundamental frequency')
plt.plot(pin_locations_exp,frequencies_exp[1,:]-frequencies_exp[1,0],'--',label='first harmonic')
plt.plot(pin_locations_exp,frequencies_exp[2,:]-frequencies_exp[2,0],'-',label='second harmonic')
plt.grid(True)
plt.xlabel('pin locations (% beam)')
plt.ylabel('$\Delta$frequency (Hz)')
plt.title('Frequency of first 3 modes')
plt.savefig('simulation_frequency',dpi=300)
plt.legend(framealpha=1,loc=2)
plt.tight_layout()

plt.tight_layout()

#%% perform FEA model updating 

# define the empty matrices 
estimated_pin_location = np.zeros((data_length))
estimated_pin_node = np.zeros((data_length),dtype=int)

# for each step in the simulated data
for i_step in range(data_length):

    # find the experimental frequencies 
    step_freqs = frequencies_exp[0:1,i_step]

    #  develop a set of unique system parameters based on the PDF centered  around the last pin location
    inputs = np.zeros((particle_number),dtype=int)
    inputs_temp = np.zeros((1),dtype=int)
    i=0
    while i < particle_number:
        inputs_temp =  int(np.random.randn(1)*step_pin_standard_dev+step_pin_node+0.5) # the 0.5 sets the int to the middle
        if inputs_temp <= 0:
            inputs_temp=1
        elif inputs_temp >= beam_node_num_FEA-2:
            inputs_temp=beam_node_num_FEA-2
        if any((inputs[:]==inputs_temp)) == False:
            inputs[i] = inputs_temp
            i=i+1        
    
    # send the model parameter to the generalized eigenvalue problem
    freqencies_FEA = np.zeros((particle_number,5))
    for i in range(particle_number):
        freqencies_FEA[i,:] = FEA.frequency_modes(mass_on=False, beam_node_num = beam_node_num_FEA, pin_node =int(inputs[i]))

    # for the first frequency, find the best match and adjust the distributions 
    step_estimated_input = np.argmin(np.abs(freqencies_FEA[:,0]-step_freqs[0]))
    
    # update the pin location 
    step_pin_node = inputs[step_estimated_input]
    
    # save results and print them to console
    estimated_pin_node[i_step] = np.copy(inputs[step_estimated_input])
    estimated_pin_location[i_step] = pin_locations_FEA[estimated_pin_node[i_step]]
    print('estimated beam location '+str(np.round(estimated_pin_location[i_step],3))+'%; and node '+str(estimated_pin_node[i_step]))

#%% Plot the results

plt.figure(figsize=(6,2.5))
plt.plot(pin_locations_exp,'-',label='truth')
plt.plot(estimated_pin_location,'--',label='estimated')
plt.title( str(beam_node_num_FEA) +' nodes; '+ str(particle_number) +' particles; '+ str(noise_level) + '% signal noise; and '+ str(step_pin_standard_dev) + ' standard deviation') 
plt.grid('on')
plt.legend(framealpha=1,loc=4,fontsize=9)
plt.xlabel('step number (#)')
plt.ylabel('roller location (% beam)')
plt.tight_layout()

plt.savefig('simulation_4.png',dpi=300)
  
    










