# In this file, we define all parameters for the GHZ state preparation in case of a spin chain with nearest neighbor interactions:

import numpy as np
M = 3 # Number of Spins
N = 2**M # Hilbert space dimension

force_mag = 2*np.pi*0.5 # (500 Mhz)
wxmax = 2*np.pi*0.5 # (500 Mhz)
wymax = 2*np.pi*0.5 # (500 Mhz)
J = 2*np.pi*0.1 #(100 MHz)


# overall time (2N) ns, simulated with 10N time steps
max_episode_steps = 10*(M)

n_substeps = 20 # substeps in ODESolver (2)
dt = 2*M/(n_substeps*max_episode_steps)  # time step

gamma = 1.0
