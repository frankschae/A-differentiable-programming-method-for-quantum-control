"""
    Differentiable Programming Quantum Control
    Example: Preparation of GHZ states in a spin chain
"""
import os, sys
sys.path.append('..')

# import common parameters
import parameters_spinchain

import numpy as np
import qutip as qu
import argparse
import pickle # to store and visualize output
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

# reproducibility is good
seed = 4
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.RandomState(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).contiguous().view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def list_kronecker(oplist, nsite):
    if nsite == 1:
        return oplist[0]
    elif nsite > 1:
        out = kronecker(oplist[0], oplist[1])
        for op in oplist[2:]:
            out = kronecker(out, op)
        return out
    else:
        print('nsite must be Integer and >=1')

def generate_drive(drive_strengths, drive):
    return torch.einsum('bi,ijk->bjk', drive_strengths, drive)

def Hamiltonian(nsite, J):
    # nsite is the number sites (spins)

    si = torch.eye(2, dtype=torch.float, device=device)
    sx = torch.tensor([[0,1],[1,0]], dtype=torch.float, device=device)
    sz = torch.tensor([[1,0],[0,-1]], dtype=torch.float, device=device)

    # Imaginary part
    sy = torch.tensor([[0,-1],[1,0]], dtype=torch.float, device=device)

    # construct operator lists
    sz_list = []
    sx_list = []
    sy_list = []


    for n in range(nsite):
        op_list = []
        for m in range(nsite):
            op_list.append(si)

        op_list[n] = sz
        sz_list.append(list_kronecker(op_list, nsite))

        op_list[n] = sx
        sx_list.append(list_kronecker(op_list, nsite))

        op_list[n] = sy
        sy_list.append(list_kronecker(op_list, nsite))
        #print(sx_list)
    # construct the hamiltonian
    Hzz = 0

    # interaction terms
    for n in range(nsite-1):
        Hzz += J*torch.matmul(sz_list[n], sz_list[n+1])

    # PBC
    # Hzz += 0.25 * self.V * torch.matmul(sz_list[nsite-1], sz_list[0])

    # sigma_x control terms
    Hx = torch.stack(sx_list)
    # sigma_y control terms
    Hy = torch.stack(sy_list)
    #print(Hzz)
    return Hzz, Hx, Hy

class PredCorrNetwork(nn.Module):
    def __init__(self, dim, nsite, n_par, target_state):
        super().__init__()

        self.dim = dim
        self.n_par = n_par
        self.nsite = nsite

        # parameters
        self.n_steps = parameters_spinchain.max_episode_steps
        self.n_substeps = parameters_spinchain.n_substeps
        self.dt = parameters_spinchain.dt
        self.gamma = parameters_spinchain.gamma
        self.force_mag = parameters_spinchain.force_mag
        print("n_steps = {}, n_substeps = {}, d_t = {}".format(self.n_steps, self.n_substeps, self.dt))

        # loss hyperparameters
        self.C1 = 0.00011  # evolution state fidelity
        # M=3: 0.00017731996539114796, M=4: 0.0001145, M=5: 0.0002, M=6: 0.0003
        self.C2 = 4.1e-06 # action amplitudes
        # M=3: 4.2277376742960046e-05, M=4: 0.018371224000611523, M=5: 1.7e-06, M=6: 1.73e-06
        self.C3 = 3.6e-07 # final state fidelity
        # M=3: 8.08065643998681e-06, M=4: 1.4085330915402378e-05, M=5: 0.13, M=6: 0.15

        # target states
        self.target_x = torch.as_tensor(np.real(target_state), dtype=torch.float, device=device).view(1,1,dim)
        self.target_y = torch.as_tensor(np.imag(target_state), dtype=torch.float, device=device).view(1,1,dim)


        # Hamiltonians
        Hzz, Hx, Hy = Hamiltonian(nsite, parameters_spinchain.J)

        self.H_0_dt = Hzz*self.dt

        self.H_1_dt = Hx*self.dt

        self.H_2_dt = Hy*self.dt
        self.Hx = Hx #for noise
        # Ground state values for initial state
        gs1 = torch.diag(self.H_0_dt)[:int(self.dim/2)]
        # gs2 = torch.diag(model.H_0_dt)[int(dim/2):]
        # print(dim)
        self.gsindex = torch.argmin(gs1)
        # print(dim-1-torch.argmin(gs1))
        # print(int(dim/2)+torch.argmin(gs2))

        # the network
        self.LS = 256
        #  M=3: 512 ,M=4: 256, M=5: 128, M=6:256
        self.LA = 64
        #  M=3: 16, M=4: 64, M=5: 32, M=6:64
        self.LC = 256
        #  M=3: 32, M=4: 256, M=5: 128, M=6:256


        layers_state = [
            nn.Linear(2*self.dim, self.LS),
        #    nn.Linear(self.LS, self.LS),# for M>4
            nn.Linear(self.LS, self.LC)
            ]

        layers_action = [
            nn.Linear(2*self.nsite, self.LA),
        #    nn.Linear(self.LA, self.LA),# for M>4
            nn.Linear(self.LA, self.LC)
            ]

        layers_combine =[
            nn.Linear(self.LC, self.LC),
        #    nn.Linear(self.LC, self.LC), # for M>4
            nn.Linear(self.LC, 2*self.nsite, bias=True)
            # 2*nsite outputs: nsite outpurs for Hx and Hy, respectively.
        ]


        # Activation functions
        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()

        self.actfunc = self.relu

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


    def Heun_complex(self, x, y, Hx_dt, Hy_dt):
        f_x, f_y = torch.matmul(Hy_dt, x)+torch.matmul(Hx_dt, y), -torch.matmul(Hx_dt, x)+torch.matmul(Hy_dt, y)
        x_tilde, y_tilde = x+f_x, y+f_y
        x, y = x+0.5*(torch.matmul(Hy_dt, x_tilde)+torch.matmul(Hx_dt, y_tilde)+f_x) , y+0.5* (-torch.matmul(Hx_dt, x_tilde)+torch.matmul(Hy_dt, y_tilde)+f_y)
        return x, y


    def forward(self, psi_x, psi_y):
        # reshape to broadcast in matmul
        x, y = psi_x.view(self.n_par, self.dim, 1), psi_y.view(self.n_par, self.dim, 1)
        alpha = torch.zeros(self.n_par, 1, 2*self.nsite, device=device)

        loss = torch.zeros(self.n_par, device=device)
        fidelity_store = torch.zeros(self.n_steps, self.n_par, device=device)
        last_action_store = torch.zeros(2, self.n_steps, self.n_par, self.nsite, device=device)

        for j in range(self.n_steps):
            input = torch.cat((x, y), 1).transpose(1,2)
            dalpha1 = self.net_state(input)

            dalpha2 = self.net_action(alpha/self.force_mag) #+ alpha/self.force_mag

            dalpha = self.net_combine(dalpha1 + dalpha2)
            alpha = self.force_mag*F.softsign(dalpha)

            alpha = torch.clamp(alpha, min=-self.force_mag, max=self.force_mag)

            alphax = alpha[:, 0, :self.nsite] # Dimension (batchsize, nsite)
            alphay = alpha[:, 0, self.nsite:] # Dimension (batchsize, nsite)

            for _ in range(self.n_substeps):
                # H_Re and H_Im have dimensions (n_par, dim, dim)
                H_Re = self.H_0_dt+generate_drive(alphax, self.H_1_dt)
                H_Im = generate_drive(alphay, self.H_2_dt)

                x, y = self.Heun_complex(x, y, H_Re, H_Im)

            fidelity = (torch.matmul(self.target_x, x)**2 + torch.matmul(self.target_x, y)**2).squeeze()

            loss += self.C1*self.gamma**j*(1-fidelity) # add state infidelity
            # punish large actions
            abs_alpha = torch.mean(alpha**2, dim=2).squeeze()
            #print(abs_alpha.shape)
            loss += self.C2*abs_alpha

            # feed storage
            fidelity_store[j] = fidelity
            last_action_store[0, j] = alphax
            last_action_store[1, j] = alphay

        psi_x, psi_y = x.view(self.n_par, self.dim), y.view(self.n_par, self.dim)

        loss += self.C3*(1-fidelity_store[-1])
        loss = loss.mean()#/self.n_steps
        return psi_x, psi_y, loss, fidelity_store, last_action_store

def render(axes, state, fidelities_mean, fidelities_std, last_actions_mean):
    global dim
    trange = np.arange(parameters_spinchain.max_episode_steps)
    x, y = state

    # clear axis of plot
    axes[0].cla()
    axes[1].cla()
    axes[2].cla()

    # plot the Fock distribution (maybe add -0.5 as in the qutip tutorial)
    plt1 = axes[0].bar(np.arange(0, dim), x**2+y**2, color='orange')
    axes[0].set_xlim([0-0.5, dim-0.5])
    axes[0].set_ylim([0, 1.0])


    plt2 = axes[1].plot(trange, last_actions_mean[0], color='blue', label='x controls')
    plt2 = axes[1].plot(trange, last_actions_mean[1], color='red', label='y controls')
    axes[1].set_xlim(0, parameters_spinchain.max_episode_steps)
    axes[1].set_ylim(-parameters_spinchain.force_mag, parameters_spinchain.force_mag)

    plt3 = axes[2].plot(trange, fidelities_mean, color='red')
    axes[2].fill_between(trange, fidelities_mean-fidelities_std, fidelities_mean+fidelities_std, alpha=0.5)
    axes[2].set_xlim(0, parameters_spinchain.max_episode_steps)
    axes[2].set_ylim(0.0, 1.0)

    axes[0].set_title(r'$|C|^2$');
    axes[1].set_title("Last actions");
    axes[2].set_title("Fidelities");

def create_init_state(noise_factor):
    global n_par, dim

    gsindex = model.gsindex
    #bool = torch.randint(0, 2, (n_par,))*2-1
    bool = torch.randint(0, 2, (n_par,))
    gsindex = gsindex + bool*(gsindex+nsite%2)
    #print(gsindex)
    psi_x, psi_y = torch.zeros((n_par,dim), device=device), torch.zeros((n_par,dim), device=device)
    psi_x[range(n_par), gsindex], psi_y[range(n_par), gsindex] = 1.0, 0.0

    mask = (torch.empty(nsite, n_par, device=device).uniform_(0, 1)).ge(1-noise_factor)

    for i in range(nsite):
        #print(i)
        #print(mask[i])
        for b in range(n_par):
            if mask[i,b]:
                psi_x[b] = torch.matmul(model.Hx[i], psi_x[b])

    return psi_x, psi_y



def train(epoch, noise_factor, optimizer, scheduler):

    # create the intial state
    psi_x, psi_y = create_init_state(noise_factor)

    with Timer('Model forward'):
        psi_x, psi_y, loss, fidelity_store, last_action_store = model.forward(psi_x, psi_y)

    with Timer('Backward'):
        optimizer.zero_grad()
        loss.backward()

    with Timer('Optimizer step'):
        optimizer.step()
        scheduler.step(loss)

    with torch.no_grad():
        if args.render == True and epoch % args.render_every == 0:
            # first plot is somehow not drawn, check this!
            #print('enter')
            psi_x_np = psi_x.cpu().detach().numpy()
            psi_y_np = psi_y.cpu().detach().numpy()
            fidelities_mean = fidelity_store.mean(dim=1).cpu().detach().numpy()
            fidelities_std = fidelity_store.std(dim=1).cpu().detach().numpy()
            last_actions_mean = last_action_store.mean(dim=2).cpu().detach().numpy()

            render(axes, (psi_x_np[0], psi_y_np[0]), fidelities_mean, fidelities_std, last_actions_mean)
            fig.tight_layout()
            plt.pause(0.05)
            plt.draw()
        #print(fidelity_store)
        print("# of epoch :{}, Loss = {}, mean norm = {}".format(epoch, loss, (psi_x**2 + psi_y**2).sum(dim=1).mean()))
        print('')
        # store performance trajectory
        pickle.dump([epoch, loss, fidelity_store[-1].mean()], fw)


@torch.no_grad()
def eval(epoch, noise_factor):
    # create the intial state
    psi_x, psi_y = create_init_state(noise_factor)

    psi_x, psi_y, loss, fidelity_store, last_action_store = model.forward(psi_x, psi_y)
    #print((x**2 + y**2))

    psi_x_np = psi_x.cpu().detach().numpy()
    psi_y_np = psi_y.cpu().detach().numpy()
    fidelities_mean = fidelity_store.mean(dim=1).cpu().detach().numpy()
    fidelities_std = fidelity_store.std(dim=1).cpu().detach().numpy()
    last_actions_mean = last_action_store.mean(dim=2).cpu().detach().numpy()

    print("# of epoch :{}, Loss = {}, mean norm = {}, Final state F = {}".format(epoch, loss, (psi_x**2 + psi_y**2).sum(dim=1).mean(), fidelity_store[-1].mean()))
    print('')


if __name__ == '__main__':
    # parse terminal input
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--optimizer', type=str, default='ADAM',
         help="Stochastic Gradient Descent (SGD), Adam (ADAM)")
    parser.add_argument('-r', '--render', type=bool, default=True,
         help="Should output be rendered?")
    parser.add_argument('-re', '--render_every', type=int, default=25,
         help="How often do you want to see the states rendered?")
    parser.add_argument('-e', '--epochs', type=int, default=2000, # 2000
         help="How many epochs the network is trained")
    args = parser.parse_args()

    dim = parameters_spinchain.N
    nsite = parameters_spinchain.M
    n_par = 512 #M=3: 256, M=4: 64, M=5: 16

    # GHZ state
    target_state = qu.ghz_state(nsite).full()


    # initialize figures to render
    if args.render == True: fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    model = PredCorrNetwork(dim, nsite, n_par, target_state)
    print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))


    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(),  lr=1e-6)
    elif args.optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr= 0.0007,)
    else:
        print("ERROR: optimizer not implemented. Choose between SGD, ADAM")
        exit()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=400000, factor=0.5, verbose=True)


    noise_factor = 0.1

    outputFile = '../data/spinchain-'+str(nsite)+'.data'
    fw = open(outputFile, 'wb')

    for epoch in range(args.epochs):
        train(epoch, noise_factor, optimizer, scheduler)
        if epoch % 25 == 0: eval(epoch, 0.0)


    fw.close()

    # store final trajectory
    filename = '../data/Fig-chain-'+str(nsite)+'-'+str(model.C1)+'-'+str(model.C2)+'-'+str(model.C3)+'.pdf'
    fig.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)

    # store final network state for serialization
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '../data/model-spinchain-'+str(nsite)+'.pth')

    exit()
