
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

class results_store:
    q1_sb_no_rtg_dsa = []

def set_plot_env(iterations, rewards_dict, exp_name):

    plt.figure(figsize=(10,5))
    style = "whitegrid"
    sns.set_theme(style=style) # background color
    ax = plt.gca()

    # rewards_dist {'name':[], 'rewards]':[]}

    ax.legend(loc='center right')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title('return of ' + exp_name +' experiment')
    ax.set_xlim([-0.5,10])

    exp_dir = 'plots/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    plt.savefig(fname=exp_dir + 'figure-2_' + exp_name + '.png', format='png')


def plot_reward(ax, rewards, name, color):
    rewards = np.array(rewards)
    iterations = np.arange(rewards.shape[0])
    ax.plot(iterations, rewards, color=color, label=name)


