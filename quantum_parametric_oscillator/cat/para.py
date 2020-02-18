"""
    Differentiable Programming Quantum Control
    Example: Preparation of cat states in a quantum parametric oscillator
"""
import os, sys
sys.path.append('..')

# import common parameters
import parameters_para

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

#muting matplotlib comments
#import logging
#logging.getLogger("imported_module").setLevel(logging.WARNING)

# reproducibility is good
seed=200
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.RandomState(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))



class PredCorrNetwork(nn.Module):
    def __init__(self, dim, n_par, target_state, pretrained=False):
        super().__init__()

        self.dim = dim
        self.n_par = n_par

        # parameters
        self.n_steps = parameters_para.max_episode_steps
        self.n_substeps = parameters_para.n_substeps
        self.dt = parameters_para.dt
        self.gamma = parameters_para.gamma
        self.force_mag = parameters_para.force_mag

        print("n_steps = {}, n_substeps = {}, d_t = {}".format(self.n_steps, self.n_substeps, self.dt))

        # loss hyperparameters
        self.C1 = 0.8 # evolution state fidelity
        self.C2 = 0.01 # action amplitudes
        self.C3 = 1*self.n_substeps # final state fidelity
        self.C4 = 0.0 # steep grad punishment

        # target states
        self.target_x = torch.as_tensor(np.real(target_state), dtype=torch.float, device=device).view(1,1,dim)
        self.target_y = torch.as_tensor(np.imag(target_state), dtype=torch.float, device=device).view(1,1,dim)


        # Hamiltonians
        a = qu.destroy(self.dim)
        H_0 = parameters_para.Kerr*(a.dag()**2)*a**2 - parameters_para.pump*(a**2+a.dag()**2)

        H_1 = (a+a.dag())
        # H_2 = (a.dag()*a)

        self.H_0_dt = np.real(H_0.full())*self.dt
        self.H_0_dt = torch.as_tensor(self.H_0_dt, dtype=torch.float, device=device)

        self.H_1_dt = np.real(H_1.full())*self.dt
        self.H_1_dt = torch.as_tensor(self.H_1_dt, dtype=torch.float, device=device)

        # self.H_2_dt = np.real(H_2.full())*self.dt
        # self.H_2_dt = torch.as_tensor(self.H_2_dt, dtype=torch.float, device=device)

        self.n_operator = torch.arange(self.dim, dtype=torch.float, device=device)

        # the network
        layers_state = [
            nn.Linear(2*dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 64),
            ]

        layers_action = [
            nn.Linear(1, 128),
            nn.Linear(128, 64)
            ]

        layers_combine =[
            nn.Linear(64, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1, bias=True)
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
            # add maybe activation function here
            self.net_combine.extend([layer, self.actfunc])
        self.net_combine.pop()
        self.net_combine = nn.Sequential(*self.net_combine).to(device)

        if pretrained:
            #https://pytorch.org/tutorials/beginner/saving_loading_models.html
            #https://pytorch.org/docs/master/notes/serialization.html
            checkpoint = torch.load('../data/para_model.pth')
            self.net_state.load_state_dict(checkpoint['model_state_dict'])
            self.net_action.load_state_dict(checkpoint['model_action_dict'])

            self.net_state.train()
            self.net_action.train()
            print("---")
            print("Pretrained network is used!")
            print("---")

    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def Heun(self,x,y,H_dt):
        f_x, f_y = torch.matmul(H_dt,y), - torch.matmul(H_dt,x)
        x_tilde, y_tilde = x + f_x, y + f_y
        x, y = x + 0.5* (torch.matmul(H_dt,y_tilde) + f_x) , y + 0.5* (-torch.matmul(H_dt,x_tilde) + f_y)
        return x, y


    def forward(self, psi_x, psi_y):
        # reshape to broadcast in matmul
        x, y = psi_x.view(self.n_par, self.dim, 1), psi_y.view(self.n_par, self.dim, 1)
        alpha = torch.zeros(self.n_par,1 , 1, device=device)

        loss = torch.zeros(self.n_par, device=device)
        fidelity_store = torch.zeros(self.n_steps, self.n_par, device=device)
        last_action_store = torch.zeros(1, self.n_steps, self.n_par, device=device)
        n_store = torch.zeros(self.n_steps, self.n_par, device=device)

        for j in range(self.n_steps):
            input = torch.cat((x, y), 1).transpose(1,2)
            dalpha1 = self.net_state(input)
            dalpha2 = self.net_action(alpha) #+ alpha/self.force_mag

            alpha = self.net_combine(dalpha1 + dalpha2)
            alpha = torch.clamp(alpha, min=-self.force_mag, max=self.force_mag)

            alpha1 = alpha[:,:, 0].unsqueeze(-1)
            # alpha2 = alpha[:,:, 1].unsqueeze(-1)

            for _ in range(self.n_substeps):
                H = self.H_0_dt+alpha1*self.H_1_dt#+alpha2*self.H_2_dt# has dimensions (n_par, dim, dim)
                x, y = self.Heun(x, y, H)

            fidelity = torch.matmul(self.target_x,x)**2 + torch.matmul(self.target_y,y)**2 + torch.matmul(self.target_y,x)**2+ torch.matmul(self.target_x,y)**2+2*torch.matmul(self.target_x,x)*torch.matmul(self.target_y,y)-2*torch.matmul(self.target_x,y)*torch.matmul(self.target_y,x)


            mean_n = torch.einsum("b, abc->a", self.n_operator, x**2+y**2)

            alpha1 = alpha1.squeeze()
            # alpha2 = alpha2.squeeze()
            loss += self.C1*self.gamma**j*(1-fidelity[:,0,0]) # add state infidelity Loss

            abs_alpha = abs(alpha1)# + abs(alpha2)
            loss += self.gamma**j*self.C2*abs_alpha
            #punish large gradients
            gradients=abs(alpha1-last_action_store[0, j-1])#+abs(alpha2-last_action_store[1, j-1])
            loss += self.C4*gradients

            # feed storage
            fidelity_store[j] = fidelity[:,0,0]
            last_action_store[0, j] = alpha1
            # last_action_store[1, j] = alpha2
            n_store[j]=mean_n

        psi_x, psi_y = x.view(self.n_par, self.dim), y.view(self.n_par, self.dim)

        loss += self.C3*(1-fidelity_store[-1])
        loss = loss.mean()#/self.n_steps
        return psi_x, psi_y, loss, fidelity_store, n_store, last_action_store

def render(axes, state, fidelities_mean, fidelities_std, last_actions_mean, n_store_mean, n_store_std):
    global dim
    trange = np.arange(parameters_para.max_episode_steps)
    x, y = state


    # clear axis of plot
    axes[0].cla()
    axes[1].cla()
    axes[2].cla()
    axes[3].cla()
    axes[4].cla()

    # plot the Fock distribution (maybe add -0.5 as in the qutip tutorial)
    plt1 = axes[0].bar(np.arange(0, dim), x**2+y**2, color='orange')
    axes[0].set_xlim([0-0.5, dim-0.5])
    axes[0].set_ylim([0, 1.0])

    # plot the Wigner function graph
    xvec = np.linspace(-6, 6, 20)
    psi_f=qu.Qobj(x[:] + 1j* y[:])
    W = qu.wigner(psi_f, xvec, xvec)
    wmap = qu.wigner_cmap(W)  # Generate Wigner colormap

    wlim = abs(W).max()
    cmap = cm.get_cmap('RdBu')
    plt2 = axes[1].contourf(xvec, xvec, W, 20,  norm=mpl.colors.Normalize(-wlim, wlim), cmap=cmap)


    plt3 = axes[2].plot(trange, last_actions_mean[0], color='blue', label='x controls')
    # plt3 = axes[2].plot(trange, last_actions_mean[1], color='red', label='y controls')
    axes[2].set_xlim(0, parameters_para.max_episode_steps)
    # axes[2].set_ylim(-parameters_para.force_mag, parameters_para.force_mag)

    plt5 = axes[3].plot(trange, fidelities_mean, color='red')
    axes[3].fill_between(trange, fidelities_mean-fidelities_std, fidelities_mean+fidelities_std, alpha=0.5)
    axes[3].set_xlim(0, parameters_para.max_episode_steps)
    axes[3].set_ylim(0.0, 1.0)

    plt6 = axes[4].plot(trange, n_store_mean, color='black')
    axes[4].fill_between(trange, n_store_mean-n_store_std, n_store_mean+n_store_std, alpha=0.3, color='black')
    axes[4].set_xlim(0, parameters_para.max_episode_steps)
    # axes[4].set_ylim(0.0, 1.0)

    axes[0].set_title(r'$|C|^2$');
    axes[1].set_title("Wigner");
    axes[2].set_title("u_x");
    axes[3].set_title("Fidelities");
    axes[4].set_title("<n>");



def create_init_state(epoch,  noise_factor=0.3):
    global n_par, dim

    psi_x, psi_y = np.zeros((n_par,dim)), np.zeros((n_par,dim))
    psi_x[:, 0], psi_y[:, 0] = 1, 0
    psi_x += noise_factor*np.random.randn(n_par,dim)*np.exp(-0.3*(np.linspace(0,dim-1,dim).reshape(1,dim)))
    psi_y += noise_factor*np.random.randn(n_par,dim)*np.exp(-0.3*(np.linspace(0,dim-1,dim).reshape(1,dim)))
    # reshape to broadcast
    norm = np.sqrt((psi_x**2+psi_y**2).sum(axis=1)).reshape(-1,1)
    #print(norm)
    psi_x, psi_y=psi_x/norm, psi_y/norm
    psi_x, psi_y = torch.from_numpy(psi_x).float().to(device), torch.from_numpy(psi_y).float().to(device)
    return psi_x, psi_y

def train(epoch, noise_factor, optimizer, scheduler):

    # create the intial state
    psi_x, psi_y = create_init_state(epoch, noise_factor)


    with Timer('Model forward'):
        psi_x, psi_y, loss, fidelity_store, n_store, last_action_store = model.forward(psi_x, psi_y)

    with Timer('Backward'):
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 40)

    with Timer('Optimizer step'):
        optimizer.step()
        # scheduler.step(loss)

    with torch.no_grad():

        psi_x_np = psi_x.cpu().detach().numpy()
        psi_y_np = psi_y.cpu().detach().numpy()
        fidelities_mean = fidelity_store.mean(dim=1).cpu().detach().numpy()
        fidelities_std = fidelity_store.std(dim=1).cpu().detach().numpy()
        last_actions_mean = last_action_store.mean(dim=2).cpu().detach().numpy()
        n_store_mean= n_store.mean(dim=1).cpu().detach().numpy()
        n_store_std = n_store.std(dim=1).cpu().detach().numpy()

        if args.render == True and epoch % args.render_every == 0:
            render(axes, (psi_x_np[0], psi_y_np[0]), fidelities_mean, fidelities_std, last_actions_mean,n_store_mean,n_store_std)
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
    psi_x, psi_y = create_init_state(epoch, noise_factor) # eval with no noise_factor

    psi_x, psi_y, loss, fidelity_store, n_store, last_action_store = model.forward(psi_x, psi_y)
    #print((x**2 + y**2))

    psi_x_np = psi_x.cpu().detach().numpy()
    psi_y_np = psi_y.cpu().detach().numpy()
    fidelities_mean = fidelity_store.mean(dim=1).cpu().detach().numpy()
    fidelities_std = fidelity_store.std(dim=1).cpu().detach().numpy()
    last_actions_mean = last_action_store.mean(dim=2).cpu().detach().numpy()

    with torch.no_grad():
        if args.render == True and epoch % args.render_every == 0:
            # first plot is somehow not drawn, check this!
            #print('enter')
            psi_x_np = psi_x.cpu().detach().numpy()
            psi_y_np = psi_y.cpu().detach().numpy()
            fidelities_mean = fidelity_store.mean(dim=1).cpu().detach().numpy()
            fidelities_std = fidelity_store.std(dim=1).cpu().detach().numpy()
            last_actions_mean = last_action_store.mean(dim=2).cpu().detach().numpy()
            n_store_mean= n_store.mean(dim=1).cpu().detach().numpy()
            n_store_std = n_store.std(dim=1).cpu().detach().numpy()

            render(axes, (psi_x_np[0], psi_y_np[0]), fidelities_mean, fidelities_std, last_actions_mean,n_store_mean,n_store_std)
            fig.tight_layout()
            plt.pause(0.5)
            plt.draw()
        #print(fidelity_store)
        print("# of epoch :{}, Loss = {}, mean norm = {}".format(epoch, loss, (psi_x**2 + psi_y**2).sum(dim=1).mean()))
        print('')

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
    parser.add_argument('-e', '--epochs', type=int, default=1000,
         help="How many epochs the network is trained")
    args = parser.parse_args()

    dim = parameters_para.N
    n_par = 64 #M=3: 256, M=4: 64, M=5: 16

    # target state
    target_a=np.sqrt(4)
    target_state = 1/np.sqrt(2)*(qu.coherent(dim,target_a)+qu.coherent(dim,-target_a)) # eigencat


    # initialize figures to render
    if args.render == True: fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    model = PredCorrNetwork(dim, n_par, target_state)
    print(model)
    # print('Check Biases:')
    # print(model.net_action[-1].bias)
    print('--------------------------------')
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))



    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(),  lr=1e-6)
    elif args.optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=0.00004,eps=1e-8)
    else:
        print("ERROR: optimizer not implemented. Choose between SGD, ADAM")
        exit()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, verbose=True)

    noise_factor = 0.4


    outputFile = '../data/para-'+str(args.epochs)+'-'+str(model.C1)+'-'+str(model.C2)+'-'+str(model.C3)+'.data'
    fw = open(outputFile, 'wb')

    for epoch in range(args.epochs):
        train(epoch, noise_factor, optimizer, scheduler)
        if epoch % args.render_every == 0:
            filename = '../data/Fig-para-noise-'+str(epoch)+'.pdf'
            fig.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                metadata=None)

            eval(epoch, 0.0)
            filename = '../data/Fig-para-no-noise-'+str(epoch)+'.pdf'
            fig.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                metadata=None)


    fw.close()

    # store final trajectory
    filename = '../data/Fig-para-final.pdf'
    fig.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)

    # store final network state for serialization
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '../data/para_model.pth')

exit()
