"""All drawing util function"""

import matplotlib.pyplot as plt
import seaborn as sns
from q_addicted import Addicted_Q_Agent
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_Q_table(num_states, num_actions, Q_across_trials):
    """
    Plain plotting of different Q valeus across diffeernt stages
    """

    fig, ax = plt.subplots(num_states, 1, figsize=(10, 20), sharex=True)
    for state in range(num_states):
        for action in range(num_actions):
            ax[state].plot(Q_across_trials[:, state, action], label=f"Action {action}")
            ax[state].set_title(f"State {state}")
            ax[state].set_ylabel("Q-value")
            ax[state].legend()

    ax[-1].set_xlabel("Trials")
    plt.tight_layout()
    plt.show()

def q_rpe_alpha_heatmap(
    action,
    alpha,
    gamma,
    epsilon,
    num_trials,
    num_states,
    num_actions,
    initial_dopamine_surge,
    dopamine_decay_rate,
    reward_states,
    drug_reward,
    addicted,
    exploration_strategy="epsilon_greedy",
):
    """
    Heat map of q table and rpe table at differnt alphas and different actions
    TODO: make it so that don't need to call with parameter

    args:
        - action: check heat map of which action (0, 1, 2)
    """

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    for i, alpha in enumerate(alpha):
        agent = Addicted_Q_Agent(
            alpha,
            gamma,
            epsilon,
            num_trials,
            num_states,
            num_actions,
            initial_dopamine_surge,
            dopamine_decay_rate,
            reward_states,
            drug_reward,
            addicted,
            exploration_strategy,
        )

        rpe, Q_across_trials = agent.learning()

        # Plot heatmap of Q-values over trials
        sns.heatmap(
            Q_across_trials[:, :, action],
            ax=axes[i, 0],
            cmap="viridis",
            cbar_kws={"label": "Q-value"},
        )
        axes[i, 0].set_title(
            f"Q-values for Action {action} over Trials (alpha={alpha})"
        )
        axes[i, 0].set_xlabel("State")
        axes[i, 0].set_ylabel("Trial")

        # Plot heatmap of RPE over trials
        sns.heatmap(
            rpe[:, :, 0], ax=axes[i, 1], cmap="viridis", cbar_kws={"label": "RPE"}
        )
        axes[i, 1].set_title(f"RPE for Action {action} over Trials (alpha={alpha})")
        axes[i, 1].set_xlabel("State")
        axes[i, 1].set_ylabel("Trial")

    plt.tight_layout()
    plt.show()

def plot_heatmap(Q_across_trials, rpe):
    '''heatmap plotting for different q-value and rpe over trials at different stages'''
    
    for action in range(3):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.heatmap(
            Q_across_trials[:, :, action],
            ax=axes[0],
            cmap="viridis",
            cbar_kws={"label": "Q-value"},
        )
        axes[0].set_title(f"Q-values for Action {action} over Trials")
        axes[0].set_xlabel("State")
        axes[0].set_ylabel("Trial")

        sns.heatmap(
            rpe[:, :, action],
            ax=axes[1],
            cmap="viridis",
            cbar_kws={"label": "RPE"},
        )
        axes[1].set_title(f"RPE for Action {action} over Trials")
        axes[1].set_xlabel("State")
        axes[1].set_ylabel("Trial")

        plt.tight_layout()
        plt.show()

def plot_avg_rpe_action(rpe, Q_across_trials):
    """Plot average rpe and different actions over trials and stages with q-values."""

    num_actions = Q_across_trials.shape[2]
    num_states = Q_across_trials.shape[1]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(
        2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.4
    )

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(rpe.mean(axis=(1, 2)))
    ax1.set_title("Reward Prediction Error across Trials")
    ax1.set_xlabel("Trials")
    ax1.set_ylabel("Average RPE")

    # Q table plot
    for action in range(num_actions):
        ax = fig.add_subplot(gs[0, action + 1])
        for state in range(num_states):
            ax.plot(Q_across_trials[:, state, action], label=f"State {state}")
        ax.set_title(f"Q-values for Action {action}")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Q-value")
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

    # RPE plot
    for action in range(num_actions):
        ax = fig.add_subplot(gs[1, action + 1])
        for state in range(num_states):
            ax.plot(rpe[:, state, action], label=f"State {state}")
        ax.set_title(f"RPE for Action {action}")
        ax.set_xlabel("Trials")
        ax.set_ylabel("RPE")
        # ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

    plt.show()