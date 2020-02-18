"""
    Reinforcement Learning Quantum Control
    Example: Qubit

    based on:
    # https://github.com/seungeunrho/minimalRL
    # https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
"""


# miscellaneous
import numpy as np
import qutip as qu
import os, sys
sys.path.append('..')

# import common parameters from file
import parameters_qubit

# import the qubit environment
import env_qubit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

eps = np.finfo(np.float32).eps.item()

import math
from itertools import count

# visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}

plt.matplotlib.rc('font', **font)

import pickle # to store and visualize output

# check run times of indiviual steps
import time
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def log_normal(x, mu, sigma_sq):
    LogGaussian = -0.5*(torch.pow(Variable(x)-mu, 2)/(sigma_sq+eps)+torch.log(2*math.pi*(sigma_sq+eps)))
    return LogGaussian

class GAE():
    def __init__(self,  max_episode_steps, u_max, w, gamma=0.999, lam=0.97):
        # for normalization trick
        self.eps = eps

        # for visualization
        self.max_episode_steps = max_episode_steps
        self.u_max = u_max
        self.w = w

        self.gamma = gamma # discount
        self.lam = lam # lambda for generalized Advantage Estimation

        self.data = []

    def put_data(self, item):
        self.data.append(item)

    def cumsum(self, datalist):
        #data = torch.as_tensor(self.data, dtype=torch.float, device=device)

        R = 0

        returns = []
        for r in datalist[::-1]: #data = torch.flip(data, dims = (0,))
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.stack(returns)

        # normalization trick
        returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + self.eps)

        return returns

    def end_episode_and_train(self, policy):
        # self.data is a list with max_episode_steps = 150 entries
        # each entry consists of a tuple with npar-rewards and npar-logprobs
        # print(data)

        self.reward_list = [x[0] for x in self.data]
        self.logprob_list = [x[1] for x in self.data]
        self.alpha_list = [x[2] for x in self.data]
        self.fidelity_list = [x[3] for x in self.data]

        # compute rewards-to-go
        with torch.no_grad():
            rtg = self.cumsum(self.reward_list)

        log_probs = torch.stack(self.logprob_list)

        policy.train_net(log_probs, rtg)

    def visualize(self, fig, axes):

        trange = np.arange(self.max_episode_steps)

        # compute mean and standard deviations
        fidelities = torch.stack(self.fidelity_list)
        actions = torch.stack(self.alpha_list)/self.w


        fidelities_mean = fidelities.mean(dim=1).cpu().detach().numpy()
        fidelities_std = fidelities.std(dim=1).cpu().detach().numpy()
        actions_mean = actions.mean(dim=1).cpu().detach().numpy()
        actions_std = actions.std(dim=1).cpu().detach().numpy()

        # clear axis of plot
        axes[0].cla()
        axes[1].cla()

        plt0 = axes[0].plot(trange, actions_mean, color='red')
        axes[0].fill_between(trange, actions_mean-actions_std, actions_mean+actions_std, alpha=0.5)
        axes[0].set_ylim(-self.u_max/self.w, self.u_max/self.w)
        axes[0].set_xlabel(r'$n$')
        axes[0].set_xlim(0, self.max_episode_steps)
        axes[0].set_ylabel(r'$u(t_n)/\omega$')

        plt1 = axes[1].plot(trange, fidelities_mean, color='red')
        axes[1].fill_between(trange, fidelities_mean-fidelities_std, fidelities_mean+fidelities_std, alpha=0.5)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xlim(0, self.max_episode_steps)
        axes[1].set_xlabel(r'$n$')
        axes[1].set_ylabel(r'$F(t_n)$')
        fig.tight_layout()
        plt.pause(0.1)
        plt.draw()

    def clear_data(self):

        # clear data
        self.data = []
        self.reward_list = []
        self.logprob_list = []
        self.alpha_list = []


class Policy(nn.Module):
    def __init__(self, dim, learning_rate, std = 0.01):
        super(Policy, self).__init__()
        # std = 0.01*self.umax, i.e. uncertainty in units of u_max
        self.std = std
        self.data = []

        self.u_max = parameters_qubit.u_max

        # initialize network layers
        # state-aware network
        layers_state = [
            nn.Linear(2*dim, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            #nn.Linear(256, 128),
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
            nn.Linear(32, 1, bias=True) # 1 if just mu is predicted, 2 if also log(sigma) can be predicted
        ]

        # Activation functions
        # Define activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
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

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)



    def forward(self, state, last_alpha):
        # Normalize last action
        input_action = (last_alpha/self.u_max)

        dalpha1 = self.net_state(state)
        dalpha2 = self.net_action(input_action)
        dalpha = self.net_combine(dalpha1 + dalpha2)

        mu = dalpha.squeeze()

        mu = self.u_max*F.softsign(mu)
        mu = torch.clamp(mu, min=-self.u_max, max=self.u_max)

        #return mu, torch.exp(log_sigma_sq)
        return mu, torch.tensor(self.std*self.u_max)

    def train_net(self, log_probs, returns):

        policy_loss = -(log_probs*returns).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()

        # maybe add here some gradient clipping
        # alternatively register a backward hook: https://stackoverflow.com/questions/54716377/how-to-properly-do-gradient-clipping-in-pytorch
        nn.utils.clip_grad_norm_(self.parameters(), 1)

        self.optimizer.step()


def select_action(policy, last_state, last_action):
    #for continuous action spaces:
    mu, sigma_sq = policy.forward(last_state, last_action)
    var = Variable(torch.randn(mu.size())).to(device)

    alpha = (mu + sigma_sq.sqrt()*var)

    # calculate the probability
    logprob = log_normal(alpha.data, mu, sigma_sq)

    return alpha, logprob


def main(seed, epochs, C1, C2, C3, dim, u_max, w, n_par, max_episode_steps, learning_rate_policy,  gamma, lam, log_interval, render, fw, std):
    env = env_qubit.QubitEnv(seed)

    # initialize models and cache
    policy = Policy(dim, learning_rate_policy, std).to(device)
    gae = GAE(max_episode_steps, u_max, w, gamma, lam)

    # initialize figures to render
    if render == True: fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    #for epoch in count(1):
    for epoch in range(epochs):
        #print(epoch)
        last_state, last_alpha = env.reset(True, n_par)
        running_reward = 0.0
        running_fidelity = 0.0


        for t in range(1, max_episode_steps+1):
            alpha, logprob = select_action(policy, last_state, last_alpha.view(n_par, 1, 1))
            #print(alpha, logprob)
            state, fidelity, abs_alpha = env.step(alpha.view(n_par, 1, 1))
            reward = C1*fidelity - C2*abs_alpha
            if t == max_episode_steps+1:
                reward += C3*fidelity
            # compute value function for state here

            # cache rewards and probability
            gae.put_data((reward, logprob, alpha, fidelity))

            last_state = state
            last_alpha = alpha

            running_reward += reward
            running_fidelity += fidelity



        gae.end_episode_and_train(policy)

        mean_reward = running_reward.mean()/max_episode_steps
        mean_fidelity = running_fidelity.mean()/max_episode_steps

        minfidelity = fidelity.min().item()
        avg_last_action = torch.sqrt(abs_alpha).mean().item()/w
        if epoch % log_interval == 0:
            print("epoch : {}, mean/min of final state fid = {}/{}, avg fidelity: {}, avg. last_action : {}".format(epoch, fidelity.mean(), minfidelity, mean_fidelity, avg_last_action))
            if render: gae.visualize(fig, axes)

        # store epoch, mean_reward, and mean of final state fidelity
        pickle.dump([epoch, mean_reward.item(), fidelity.mean().item()], fw)

        # clear data for next iteration
        gae.clear_data()

        if minfidelity > 0.99 and mean_fidelity.item()>0.92 and avg_last_action<0.04:
            # store final trajectory
            filename = '../data/Fig-final-qubit-RL.pdf'
            fig.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                metadata=None)
            break

    return mean_fidelity.item(), fidelity.mean().item(), policy

if __name__ == '__main__':
    # reproducibility is good
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    epochs = 35000

    #Hyperparameters
    n_par = 1024 # number of episodes before updating the networks
    u_max = parameters_qubit.u_max
    w = parameters_qubit.w
    dim = parameters_qubit.N
    max_episode_steps = parameters_qubit.max_episode_steps

    gamma = parameters_qubit.gamma
    lam = 0.97 # Hyperparameter for generalized advantage estimation
    learning_rate_policy = 0.00025

    log_interval = 50
    render = True

    # percentage*force_mag = variance of the Gaussian policy
    std = 0.04

    C1 = 0.016
    C2 = 3.9e-06
    C3 = 0.0002

    # store the training data
    outputFile = '../data/qubit-RL-'+str(epochs)+'-'+str(C1)+'-'+str(C2)+'-'+str(C3)+'.data'
    fw = open(outputFile, 'wb')

    with Timer('time to train the model: '):
        last_fidelities, final_state_fidelity, model = main(
            seed, epochs,
            C1, C2, C3,
            dim, u_max, w, n_par, max_episode_steps,
            learning_rate_policy, gamma, lam,
            log_interval, render, fw, std
            )

    print(last_fidelities, final_state_fidelity)

    fw.close()

    # store final network state for serialization
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        }, '../data/model-qubit-RL-'+str(C1)+'-'+str(C2)+'-'+str(C3)+'.pth')

    exit('Finished!')
