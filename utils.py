'''All drawing util function'''

import matplotlib.pyplot as plt
import seaborn as sns
from q_addicted import addicted_q_learning
import numpy as np

def plot_Q_table(num_states, num_actions, Q_across_trials):
    '''
    Plain plotting of different Q valeus across diffeernt stages
    '''

    fig, ax = plt.subplots(num_states, 1, figsize=(10, 20), sharex=True)
    for state in range(num_states):
        for action in range(num_actions):
            ax[state].plot(Q_across_trials[:, state, action], label=f'Action {action}')
            ax[state].set_title(f'State {state}')
            ax[state].set_ylabel('Q-value')
            ax[state].legend()

    ax[-1].set_xlabel('Trials')
    plt.tight_layout()
    plt.show()

def q_rpe_alpha_heatmap(action, alpha, gamma, epsilon, num_trials,
                num_states, num_actions, initial_dopamine_surge,
                dopamine_decay_rate, reward_states, drug_reward, addicted):
    '''
    Heat map of q table and rpe table at differnt alphas and different actions
    TODO: make it so that don't need to call with parameter

    args:
        - action: check heat map of which action (0, 1, 2)
    '''

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    for i, alpha in enumerate(alpha):
        rpe, Q_across_trials = addicted_q_learning(alpha, gamma, epsilon, num_trials,
                                            num_states, num_actions, initial_dopamine_surge,
                                            dopamine_decay_rate, reward_states, drug_reward, addicted)

        # Plot heatmap of Q-values over trials
        sns.heatmap(Q_across_trials[:, :, action], ax=axes[i, 0], cmap="viridis", cbar_kws={'label': 'Q-value'})
        axes[i, 0].set_title(f'Q-values for Action {action} over Trials (alpha={alpha})')
        axes[i, 0].set_xlabel('State')
        axes[i, 0].set_ylabel('Trial')

        # Plot heatmap of RPE over trials
        sns.heatmap(rpe[:, :, 0], ax=axes[i, 1], cmap="viridis", cbar_kws={'label': 'RPE'})
        axes[i, 1].set_title(f'RPE for Action {action} over Trials (alpha={alpha})')
        axes[i, 1].set_xlabel('State')
        axes[i, 1].set_ylabel('Trial')

    plt.tight_layout()
    plt.show()