
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

class results_store:
    q1_sb_no_rtg_dsa = [29.0, 27.466667, 48.555557, 48.555557, 56.125, 41.7, 38.636364, 40.6, 36.916668, 40.1, 50.444443, 71.333336, 77.833336, 50.875, 55.5, 70.0, 41.5, 40.0, 37.272728, 35.583332, 40.9, 38.583332, 35.166668, 51.375, 51.0, 71.14286, 49.22222, 58.625, 74.166664, 63.142857, 59.285713, 51.0, 91.6, 54.5, 64.71429, 74.666664, 59.0, 57.125, 52.75, 43.0, 36.5, 34.333332, 30.642857, 26.933332, 26.0625, 30.071428, 27.066668, 26.1875, 28.266666, 26.933332, 27.266666, 27.533333, 25.75, 26.1875, 24.117647, 27.2, 27.0625, 30.76923, 33.23077, 36.545456, 33.666668, 40.7, 37.5, 45.444443, 38.0, 58.285713, 54.0, 58.142857, 37.18182, 51.25, 48.333332, 50.75, 60.142857, 90.8, 54.0, 51.5, 48.333332, 50.375, 82.8, 87.2, 88.2, 85.2, 60.142857, 59.75, 82.0, 64.57143, 59.285713, 67.666664, 88.8, 86.0, 78.333336, 101.5, 111.0, 75.333336, 72.0, 93.8, 72.166664, 91.0, 90.8, 117.0]
    q1_sb_rtg_dsa = []
    q1_sb_rtg_na = []
    q1_lb_no_rtg_dsa = []
    q1_lb_rtg_dsa = []
    q1_lb_rtg_na = []



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


