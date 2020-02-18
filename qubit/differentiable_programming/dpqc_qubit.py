"""
    Differentiable Programming Quantum Control
    Example: Qubit
"""
import os, sys
sys.path.append('..')

# import common parameters from file
import parameters_qubit

# miscellaneous
import numpy as np
import qutip as qu
import argparse
import pickle # to store and visualize output
import time

# PyTorch utilities
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

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

# reproducibility is good
seed = 2 #(200)
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.RandomState(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# check run times of indiviual steps
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

# main class
class PredCorrNetwork(nn.Module):
    def __init__(self, dim, n_par, target_state, C1, C2, C3):
        super().__init__()

        self.dim = dim
        self.n_par = n_par

        # parameters_qubit
        self.n_steps = parameters_qubit.max_episode_steps
        self.n_substeps = parameters_qubit.n_substeps
        self.dt = parameters_qubit.dt
        self.gamma = parameters_qubit.gamma
        self.u_max = parameters_qubit.u_max
        self.w = parameters_qubit.w

        # loss hyperparameters
        self.C1 = C1  # L_F
        self.C2 = C2  # L'_amp
        self.C3 = C3  # L_FN

        # target state
        self.target_x = torch.as_tensor(np.real(target_state), dtype=torch.float, device=device).view(1,1,dim)
        self.target_y = torch.as_tensor(np.imag(target_state), dtype=torch.float, device=device).view(1,1,dim)

        # Hamiltonians
        # drift term
        H_0 = self.w/2*qu.sigmaz()
        # control term
        H_1 = qu.sigmax()

        self.H_0_dt = np.real(H_0.full())*self.dt
        self.H_0_dt = torch.as_tensor(self.H_0_dt, dtype=torch.float, device=device)

        self.H_1_dt = np.real(H_1.full())*self.dt
        self.H_1_dt = torch.as_tensor(self.H_1_dt,dtype=torch.float, device=device)


        # initialize network layers
        # state-aware network
        layers_state = [
            nn.Linear(2*dim, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            ]
        # action-aware network
        layers_action = [
            nn.Linear(1, 128),
            nn.Linear(128, 128),
            ]
        # combination-aware network
        layers_combine =[
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1, bias=True)
        ]

        # Activation functions
        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()

        # Use ReLU for the experiment
        self.actfunc = self.relu

        # predictive model
        self.net_state = []
        for layer in layers_state:
            self.net_state.extend([layer, self.actfunc])
        #self.net_state.pop()
        self.net_state = nn.Sequential(*self.net_state).to(device)

        self.net_action = []
        for layer in layers_action:
            self.net_action.extend([layer, self.actfunc])
        #self.net_action.pop()
        self.net_action = nn.Sequential(*self.net_action).to(device)

        self.net_combine = []
        for layer in layers_combine:
            self.net_combine.extend([layer, self.actfunc])
        self.net_combine.pop()
        self.net_combine = nn.Sequential(*self.net_combine).to(device)

    def Heun(self,x,y,H_dt):
        # ODE solver
        f_x, f_y = torch.matmul(H_dt,y), - torch.matmul(H_dt,x)
        x_tilde, y_tilde = x + f_x, y + f_y
        x, y = x + 0.5* (torch.matmul(H_dt,y_tilde) + f_x) , y + 0.5* (-torch.matmul(H_dt,x_tilde) + f_y)
        return x, y

    def forward(self, psi_x, psi_y):
        # reshape to broadcast in matmul
        x, y = psi_x.view(self.n_par, self.dim, 1), psi_y.view(self.n_par, self.dim, 1)
        # control field
        alpha = torch.zeros(self.n_par,1 ,1, device=device)

        loss = torch.zeros(self.n_par, device=device)
        fidelity_store = torch.zeros(self.n_steps, self.n_par, device=device)
        last_action_store = torch.zeros(self.n_steps, self.n_par, device=device)

        for j in range(self.n_steps):
            input = torch.cat((x, y), 1).transpose(1,2)
            dalpha1 = self.net_state(input)
            dalpha2 = self.net_action(alpha/self.u_max)
            dalpha = self.net_combine(dalpha1 + dalpha2)
            dalpha = F.softsign(dalpha)

            alpha = self.u_max*dalpha
            alpha = torch.clamp(alpha, min=-self.u_max, max=self.u_max)

            for _ in range(self.n_substeps):
                H = self.H_0_dt+alpha*self.H_1_dt # has dimensions (n_par, 16,16)
                x, y = self.Heun(x, y, H)

            fidelity = (torch.matmul(self.target_x, x)**2 + torch.matmul(self.target_x, y)**2).squeeze()

            alpha = alpha.squeeze()

            loss += self.C1*self.gamma**j*(1-fidelity) # add state infidelity
            # punish large actions
            abs_alpha = alpha**2
            loss += self.C2*abs_alpha

            # feed storage
            fidelity_store[j] = fidelity
            last_action_store[j] = alpha

            alpha = alpha.view(-1, 1, 1)

        psi_x, psi_y = x.view(self.n_par, self.dim), y.view(self.n_par, self.dim)

        loss += self.C3*(1-fidelity_store[-1])
        loss = loss.mean()

        return psi_x, psi_y, loss, fidelity_store, last_action_store

def render(axes, state, fidelities_mean, fidelities_std, last_actions_mean, last_actions_std):
    #global dim
    N = parameters_qubit.max_episode_steps
    trange = np.arange(N)

    # clear axis of plot
    axes[0].cla()
    axes[1].cla()

    plt0 = axes[0].plot(trange, last_actions_mean, color='red')
    axes[0].fill_between(trange, last_actions_mean-last_actions_std, last_actions_mean+last_actions_std, alpha=0.5)
    axes[0].set_ylim(-parameters_qubit.u_max/parameters_qubit.w, parameters_qubit.u_max/parameters_qubit.w)
    axes[0].set_xlabel(r'$n$')
    axes[0].set_xlim(0, N)
    axes[0].set_ylabel(r'$u(t_n)/\omega$')

    plt1 = axes[1].plot(trange, fidelities_mean, color='red')
    axes[1].fill_between(trange, fidelities_mean-fidelities_std, fidelities_mean+fidelities_std, alpha=0.5)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlim(0, N)
    axes[1].set_xlabel(r'$n$')
    axes[1].set_ylabel(r'$F(t_n)$')


def create_init_state(noise, n_par):

    dim = 2

    psi_x, psi_y = torch.zeros((n_par,dim), dtype=torch.float, device=device), torch.zeros((n_par,dim), dtype=torch.float, device=device)

    if noise:
        # Note that theta [0, 2pi] is biased towards the pols
        # theta	=	cos^(-1)(2v-1) with v on [0,1]
        theta = torch.acos(torch.zeros((n_par,), dtype=torch.float, device=device).uniform_(-1.0, 1.0))
        phi = torch.zeros((n_par,), dtype=torch.float, device=device).uniform_(0.0, 2*np.pi)

        psi_x[:, 0] += torch.cos(theta / 2) # real part of coefficient of |up>
        psi_x[:, 1] += torch.sin(theta / 2)*torch.cos(phi) # real part of coefficient of |down>

        psi_y[:, 0] += torch.zeros_like(theta)  # imag part of coefficient of |up>
        psi_y[:, 1] += torch.sin(theta / 2)*torch.sin(phi)
    else:
        psi_x[:, 0], psi_y[:, 0] = 1, 0 #1, 0 # |up>

    return psi_x, psi_y

def train(epoch, n_par, noise, axes, optimizer, scheduler, fw):

    # create the intial state
    psi_x, psi_y = create_init_state(noise, n_par)

    with Timer('Model forward'):
        psi_x, psi_y, loss, fidelity_store, last_action_store = model.forward(psi_x, psi_y)

    with Timer('Backward'):
        optimizer.zero_grad()
        loss.backward()

    with Timer('Optimizer step'):
        optimizer.step()
        #scheduler.step(loss)

    with torch.no_grad():
        if args.render == True and epoch % args.render_every == 0:
            psi_x_np = psi_x.cpu().detach().numpy()
            psi_y_np = psi_y.cpu().detach().numpy()
            fidelities_mean = fidelity_store.mean(dim=1).cpu().detach().numpy()
            fidelities_std = fidelity_store.std(dim=1).cpu().detach().numpy()
            last_actions_mean = last_action_store.mean(dim=1).cpu().detach().numpy()/parameters_qubit.w
            last_actions_std = last_action_store.std(dim=1).cpu().detach().numpy()/parameters_qubit.w
            render(axes, [psi_x_np[0], psi_y_np[0]], fidelities_mean, fidelities_std, last_actions_mean, last_actions_std)
            fig.tight_layout()
            plt.pause(0.1)
            plt.draw()

        print("# of epoch :{}, Loss = {}, mean norm = {}".format(epoch, loss, (psi_x**2 + psi_y**2).sum(dim=1).mean()))
        print('')
        # store performance trajectory
        pickle.dump([epoch, loss.detach(), fidelity_store[-1].mean().detach()], fw)

if __name__ == '__main__':
    # parse terminal input
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--optimizer', type=str, default='ADAM',
         help="Stochastic Gradient Descent (SGD), Adam (ADAM)")
    parser.add_argument('-r', '--render', type=bool, default=True,
         help="Should output be rendered?")
    parser.add_argument('-re', '--render_every', type=int, default=1,
         help="How often do you want to see the states rendered?")
    parser.add_argument('-e', '--epochs', type=int, default=400,
         help="How many epochs the network is trained")
    args = parser.parse_args()

    # Hilbert space dimension
    dim = 2
    # number of parallel simulations
    n_par = 256

    # coefficients of the loss functions
    C1 = 0.5764456575989 #0.5 #0.000310794377545843 #0.576445657598991
    C2 = 0.0026925155809219 #0.003 #0.0 #0.002692515580921919
    C3 = 3.8672613833194e-06 #4e-6 #0.0 #3.867261383319492e-06

    # learning rate
    lr =  0.0032737004109265697 # 0.0004905088071897614 # 0.0032737004109265697

    # Target state:
    # |1> = |down> = np.array([0.0, 1.0])
    target_state = np.array([0.0, 1.0])

    #tsimulation = parameters_qubit.dt*parameters_qubit.n_substeps*parameters_qubit.max_episode_steps
    #print('Max simulation time (ns): ', tsimulation)

    # initialize figures to render
    if args.render == True: fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # initialize the model
    model = PredCorrNetwork(dim, n_par, target_state, C1, C2, C3)
    print(model)

    # set the optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(),  lr=lr)
    elif args.optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        print("ERROR: optimizer not implemented. Choose between SGD, ADAM.")
        exit()

    # scheduler for lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5, verbose=True)

    # Uncertainty in the initial state (here: fully arbitrary initial state)
    noise = True

    # store the training data
    outputFile = '../data/qubit-'+str(args.epochs)+'-'+str(model.C1)+'-'+str(model.C2)+'-'+str(model.C3)+'.data'
    fw = open(outputFile, 'wb')

    # training loop
    for epoch in range(args.epochs):
        train(epoch, n_par, noise, axes, optimizer, scheduler, fw)

    fw.close()

    # store final trajectory
    filename = '../data/Fig-qubit-'+str(model.C1)+'-'+str(model.C2)+'-'+str(model.C3)+'.pdf'
    fig.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)

    # store final network state for serialization
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '../data/model-qubit-'+str(model.C1)+'-'+str(model.C2)+'-'+str(model.C3)+'.pth')

    exit('Finished!')
