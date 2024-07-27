"""All drawing util functions for figures"""

import matplotlib.pyplot as plt
import seaborn as sns
from src.q_addicted import Addicted_Q_Agent
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_heatmap_for_configs(CONFIGS):
    '''Function tha tplots for figure 1'''

    labels = ['Epsilon Greedy Addicted',
              'Boltzmann Exploration Addictted',
              'Epsilon Greedy addicted Reward',
              'Boltzmann Exploration addicted Reward']
    actions = ['Move Forward', 'Move Backward', 'Stay']
    
    agent1 = Addicted_Q_Agent(
            CONFIGS['alpha'],
            CONFIGS['gamma'],
            CONFIGS['epsilon'],
            CONFIGS['num_trials'],
            CONFIGS['num_states'],
            CONFIGS['num_actions'],
            CONFIGS['initial_dopamine_surge'],
            CONFIGS['dopamine_decay_rate'],
            CONFIGS['reward_states'],
            CONFIGS['drug_reward'],
            CONFIGS['addicted'],
            'epsilon_greedy',
            False,
            CONFIGS['addicted_reward_states'],
            CONFIGS['addicted_reward_boost']
        )
    agent2 = Addicted_Q_Agent(
            CONFIGS['alpha'],
            CONFIGS['gamma'],
            CONFIGS['epsilon'],
            CONFIGS['num_trials'],
            CONFIGS['num_states'],
            CONFIGS['num_actions'],
            CONFIGS['initial_dopamine_surge'],
            CONFIGS['dopamine_decay_rate'],
            CONFIGS['reward_states'],
            CONFIGS['drug_reward'],
            CONFIGS['addicted'],
            'boltzmann_exploration',
            False,
            CONFIGS['addicted_reward_states'],
            CONFIGS['addicted_reward_boost']
        )
    agent3 = Addicted_Q_Agent(
            CONFIGS['alpha'],
            CONFIGS['gamma'],
            CONFIGS['epsilon'],
            CONFIGS['num_trials'],
            CONFIGS['num_states'],
            CONFIGS['num_actions'],
            CONFIGS['initial_dopamine_surge'],
            CONFIGS['dopamine_decay_rate'],
            CONFIGS['reward_states'],
            CONFIGS['drug_reward'],
            CONFIGS['addicted'],
            'epsilon_greedy',
            True,
            CONFIGS['addicted_reward_states'],
            CONFIGS['addicted_reward_boost']
        )
    agent4 = Addicted_Q_Agent(
            CONFIGS['alpha'],
            CONFIGS['gamma'],
            CONFIGS['epsilon'],
            CONFIGS['num_trials'],
            CONFIGS['num_states'],
            CONFIGS['num_actions'],
            CONFIGS['initial_dopamine_surge'],
            CONFIGS['dopamine_decay_rate'],
            CONFIGS['reward_states'],
            CONFIGS['drug_reward'],
            CONFIGS['addicted'],
            'boltzmann_exploration',
            True,
            CONFIGS['addicted_reward_states'],
            CONFIGS['addicted_reward_boost']
        )
    
    agents = [agent1, agent2, agent3, agent4]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i, agent in enumerate(agents):
        _, Q_across_trials = agent.learning()

        for action_id, action in enumerate(actions):
            ax = axes[action_id, i]
            sns.heatmap(
                Q_across_trials[:, :, action_id],
                ax=ax,
                cmap="viridis",
                cbar_kws={"label": "Q-value"},
            )
            # ax.set_title(f"Action {action} - {labels[i]}")
            ax.set_xlabel("State")
            ax.set_ylabel("Trial")
        
        for i, label in enumerate(labels):
            axes[0, i].set_title(label, fontsize=16, pad=20)
        
        for j, label in enumerate(actions):
            axes[j, 0].set_ylabel(label, fontsize=16, labelpad=20)
    
    plt.tight_layout()
    plt.show()

def plot_all_rpe_avg(CONFIGS):
    '''Function for outputting figure 2'''

    all_rpe_addicted = []
    all_rpe_non_addicted = []
    strategies = ['greedy', 'epsilon_greedy', 'boltzmann_exploration']
    strategies_name = ['Greedy', 'Epsilon Greedy', 'Boltzmann Exploration']
    colors = ['r', 'g', 'b']
    
    for strategy in strategies:
        for addicted, all_rpe in zip([True, False], [all_rpe_addicted, all_rpe_non_addicted]):
            agent = Addicted_Q_Agent(
                CONFIGS['alpha'],
                CONFIGS['gamma'],
                CONFIGS['epsilon'],
                500,
                CONFIGS['num_states'],
                CONFIGS['num_actions'],
                CONFIGS['initial_dopamine_surge'],
                CONFIGS['dopamine_decay_rate'],
                CONFIGS['reward_states'],
                CONFIGS['drug_reward'],
                addicted,
                strategy,
                False,
                CONFIGS['natural_reward_states'],
                CONFIGS['natural_reward_boost']
            )
            rpe, _ = agent.learning()
            all_rpe.append(rpe)
        
    # average RPE
    average_rpe_addicted = [rpe.mean(axis=(1, 2)) for rpe in all_rpe_addicted]
    average_rpe_non_addicted = [rpe.mean(axis=(1, 2)) for rpe in all_rpe_non_addicted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # plot addicted condition
    for i, (strategy, avg_rpe, color) in enumerate(zip(strategies_name, average_rpe_addicted, colors)):
        axes[0].plot(avg_rpe, label=strategy, color=color)
    axes[0].set_xlabel("Trials")
    axes[0].set_ylabel("Average Reward Prediction Error")
    axes[0].legend()
    axes[0].set_title("Average RPE for Addicted Condition")
    
    # plot non-addicted condition
    for i, (strategy, avg_rpe, color) in enumerate(zip(strategies_name, average_rpe_non_addicted, colors)):
        axes[1].plot(avg_rpe, label=strategy, color=color)
    axes[1].set_xlabel("Trials")
    axes[1].set_ylabel("Average Reward Prediction Error")
    axes[1].legend()
    axes[1].set_title("Average RPE for Non-addicted Condition")
    
    plt.tight_layout()
    plt.show()


def plot_all_expected_visits(CONFIGS):
    '''Function for outputting figure 3'''

    agent1 = Addicted_Q_Agent(
        CONFIGS['alpha'],
        CONFIGS['gamma'],
        CONFIGS['epsilon'],
        CONFIGS['num_trials'],
        CONFIGS['num_states'],
        CONFIGS['num_actions'],
        CONFIGS['initial_dopamine_surge'],
        CONFIGS['dopamine_decay_rate'],
        CONFIGS['reward_states'],
        CONFIGS['drug_reward'],
        CONFIGS['addicted'],
        'epsilon_greedy',
        CONFIGS['if_addicted'],
        CONFIGS['addicted_reward_states'],
        CONFIGS['addicted_reward_boost']
    )
    agent2 = Addicted_Q_Agent(
        CONFIGS['alpha'],
        CONFIGS['gamma'],
        CONFIGS['epsilon'],
        CONFIGS['num_trials'],
        CONFIGS['num_states'],
        CONFIGS['num_actions'],
        CONFIGS['initial_dopamine_surge'],
        CONFIGS['dopamine_decay_rate'],
        CONFIGS['reward_states'],
        CONFIGS['drug_reward'],
        CONFIGS['addicted'],
        'boltzmann_exploration',
        CONFIGS['if_addicted'],
        CONFIGS['addicted_reward_states'],
        CONFIGS['addicted_reward_boost']
    )

    agent1.learning()
    agent2.learning()

    num_re_trials = 10000
    max_action_per_trial = 100

    avg_durations_eps = agent1.resimulate_state_durations(num_re_trials, max_action_per_trial)
    avg_durations_boltz = agent2.resimulate_state_durations(num_re_trials, max_action_per_trial)

    # random walk
    num_trials = 100
    num_steps = 1000
    avg_durations_rw = agent1.random_walk(num_trials, num_steps)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].bar(range(CONFIGS['num_states']), avg_durations_rw)
    axs[0].set_title('Random Walk Expected Visits')
    axs[0].set_xlabel('State')
    axs[0].set_ylabel('Expected Visits')
    axs[0].set_xticks(range(CONFIGS['num_states']))

    axs[1].bar(range(CONFIGS['num_states']), avg_durations_eps)
    axs[1].set_title('Trained Epsilon Greedy Agent Expected Visits')
    axs[1].set_xlabel('State')
    axs[1].set_ylabel('Expected Visits')
    axs[1].set_xticks(range(CONFIGS['num_states']))

    axs[2].bar(range(CONFIGS['num_states']), avg_durations_boltz)
    axs[2].set_title('Trained Boltzmann Exploration Agent Expected Visits')
    axs[2].set_xlabel('State')
    axs[2].set_ylabel('Expected Visits')
    axs[2].set_xticks(range(CONFIGS['num_states']))

    plt.tight_layout()
    plt.show()