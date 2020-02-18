"""
    RL Quantum Control
    Example: Qubit
"""
# miscellaneous
import numpy as np
import qutip as qu
import os, sys
sys.path.append('..')
sys.path.append('../differentiable_programming')

# PyTorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F

# import common parameters from file
import parameters_qubit

# import function to prepare random initial state on the bloch sphere
from dpqc_qubit import create_init_state

# visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}

plt.matplotlib.rc('font', **font)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class QubitEnv():
    """
    Qubit
    dt(float): integration time
    """

    def __init__(self, seed):
                # parameters_qubit
        self.dt = parameters_qubit.dt  # time step
        self.n_substeps = parameters_qubit.n_substeps # substeps in ODESolver
        self.dim = parameters_qubit.N # dimension of Hilbert space
        self.u_max = parameters_qubit.u_max
        self.w = parameters_qubit.w

        # target state

        target_state = np.array([0.0, 1.0])
        self.target_x = torch.as_tensor(np.real(target_state), dtype=torch.float, device=device).view(1,1,self.dim)
        self.target_y = torch.as_tensor(np.imag(target_state), dtype=torch.float, device=device).view(1,1,self.dim)


        # Hamiltonians
        # drift term
        H_0 = self.w/2*qu.sigmaz()
        # control term
        H_1 = qu.sigmax()

        self.H_0_dt = np.real(H_0.full())*self.dt
        self.H_0_dt = torch.as_tensor(self.H_0_dt, dtype=torch.float, device=device)

        self.H_1_dt = np.real(H_1.full())*self.dt
        self.H_1_dt = torch.as_tensor(self.H_1_dt,dtype=torch.float, device=device)


        fixed_seed = True
        if fixed_seed:
            self.seed(seed)

    def seed(self, seed=None):
        # reproducibility is good
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @torch.no_grad()
    def step(self, alpha):
        # alpha must have the form (n_par, 1, 1)

        # time step in integrator
        dt = self.dt

        alpha = torch.clamp(alpha, min=-self.u_max, max=self.u_max)

        # EOM
        x, y = self.state

        for _ in range(self.n_substeps):
            H = self.H_0_dt+alpha*self.H_1_dt # has dimensions (n_par, 16,16)
            x, y = self.Heun(x, y, H)

        # compute L(t)
        fidelity = (torch.matmul(self.target_x, x)**2 + torch.matmul(self.target_x, y)**2).squeeze()
        abs_alpha = (alpha**2).squeeze()

        # update state and last action
        self.last_alpha = alpha
        self.state = x, y

        return self._get_obs(), fidelity, abs_alpha

    @torch.no_grad()
    def reset(self, noise, n_par):#
        # create the intial state
        psi_x, psi_y = create_init_state(noise, n_par) # with batch size n_par

        self.state = psi_x.view(n_par, self.dim, 1), psi_y.view(n_par, self.dim, 1)

        self.last_alpha = torch.zeros((n_par, 1, 1), device=device)

        return self._get_obs(), self.last_alpha

    @torch.no_grad()
    def _get_obs(self):
        x, y = self.state
        return torch.cat((x, y), 1).transpose(1,2)

    @torch.no_grad()
    def Heun(self,x,y,H_dt):
        # ODE solver
        f_x, f_y = torch.matmul(H_dt,y), - torch.matmul(H_dt,x)
        x_tilde, y_tilde = x + f_x, y + f_y
        x, y = x + 0.5* (torch.matmul(H_dt,y_tilde) + f_x) , y + 0.5* (-torch.matmul(H_dt,x_tilde) + f_y)
        return x, y


if __name__ == '__main__':
    # test continuous Qubit environment
    seed = 100
    env = QubitEnv(seed)

    n_par = 3

    state, last_action = env.reset(True, n_par)

    print(state)
    for _ in range(10):
        alpha = torch.randn((n_par, 1, 1), device=device)
        state_prime, fidelity, abs_alpha = env.step(alpha)
        print(abs_alpha)
    exit()
