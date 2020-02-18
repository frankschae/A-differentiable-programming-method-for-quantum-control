# In this file, all parameters for the qubit control task are defined:

import numpy as np
N = 2

w = 2*np.pi*3.9  # (GHz)
u_max = 2*np.pi*0.3 #(300 MHz)

max_episode_steps = 150
dt = 0.001  # time step
n_substeps = 20 # substeps in ODESolver
gamma = 1.0
