'''Major model used in this study'''

#TODO: make q-agent class

import numpy as np
import random

def addicted_q_learning(alpha: float = 0.1,
                        gamma: float = 0.9,
                        epsilon: float = 0.1,
                        num_trials: int = 100,
                        num_states: int = 10,
                        num_actions: int = 3,
                        initial_dopamine_surge: int = 1,
                        dopamine_decay_rate: float = 0.09,
                        reward_states: list = [9],
                        drug_reward: int = 1,
                        addicted_agent: bool = True):
    '''
    Simulating addicted Q learning agent, using epsilon greedy policy

    action space: [0,1,2]:
        - 0 move forward, 1 move backward, 2 stay

    reward is only recieved in the drug state
        - dopamine_surge is added to act as a effect of the drug (exponentially decrease)
        - drug stage can be multiple occurances to test effects of compettition in drugs
        - can turn flags of addiction or not to switch between addicted agent and non-addicted agents
    
    args:
        - alpha: learning rate
        - gamma: decay rate
        - epsilon: exploration baclance
        - num_trials: number of trials
        - num_states: number of states
        - num_actions: number of actions
        - initial_dopamine_surge: dopamine surge initial value
        - dopamine_decay_rate: dopamine surge decay rate
        - reward_states: list object, any number of reward states, must be smaller than num_states
        - drug_reward: reward of the drug rewrad state
        - addicted: boolean argument, simulating an addicting agent of not

    return:
        - Reward prediction error over trials
        - Q table over trials
    '''

    assert len(reward_states) <= num_states, "reward states must be less than number of states"

    # initialize Q (action values) table, errors, and state_action values
    Q = np.zeros((num_states, num_actions))
    rpe = np.zeros((num_trials, num_states, num_actions))
    Q_across_trials = np.zeros((num_trials, num_states, num_actions))

    for trial in range(num_trials):
        dopamine_surge = initial_dopamine_surge * (dopamine_decay_rate ** trial)

        for state in range(num_states):
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, num_actions - 1)
            else:
                action = np.argmax(Q[state])

            if action == 0:
                next_state = min(state + 1, num_states - 1)
            elif action == 1:
                next_state = max(state - 1, 0)
            else:
                next_state = state
            
            # reward activation
            reward = drug_reward if next_state in [reward_state for reward_state in reward_states] else 0

            # update TD
            max_next_Q = np.max(Q[next_state])

            
            delta = reward + gamma * max_next_Q - Q[state, action]

            if addicted_agent:
                delta = max(reward + gamma * max_next_Q - Q[state, action] + dopamine_surge * reward, dopamine_surge * reward)
            
            # update table
            Q[state, action] = Q[state, action] + alpha * delta
            rpe[trial, state, action] = delta
            Q_across_trials[trial, state, action] = Q[state, action]

    return rpe, Q_across_trials


def simulate_state_durations(Q, num_trials=100, num_states=10):
    '''
    Re-simulate average state durations using Q-values and calculate the average duration in each state.
    In stochastic processes, average visits is a key metrics of the chain or the  process, so we try to do the same
    simulation wise here with using best learned Q
    
    args:
        - Q: Q-table
        - num_trials: number of trials
        - num_states: number of states
    
    return:
        - average duration in each state
    '''
    state_durations = np.zeros((num_trials, num_states))

    for trial in range(num_trials):
        state = np.random.randint(num_states)
        
        counter = 0
        while True:
            action = np.argmax(Q[state])
            print(action)
            if action == 0 and state < num_states - 1:
                next_state = state + 1
            elif action == 1 and state > 0:
                next_state = state - 1
            else:
                next_state = state
            
            state_durations[trial, state] += 1
            state = next_state

            counter+=1
            
            # stop condition
            if state == num_states - 1:
                break

            if counter == 10:
                break
    
    avg_durations = np.mean(state_durations, axis=0)

    return avg_durations


def random_walk(num_trials=100, num_steps=1000, num_states=10):
    '''
    Markov-chain assumption holds

    Discrete stochastic process (Markov chain) modeling using random walk
        - 50% forward, 50% backward

    args:
        - num_trials: total number of trials
        - num_steps: numebr of steps per trials
        - num_states: number of states (should be the same with TD agent)

    return:
        - average duration of each states
    '''
    
    P = np.zeros((num_states, num_states))
    
    # transition probability matrix
    for state in range(num_states):
        if state < num_states - 1:
            P[state, state + 1] = 0.5
        if state > 0:
            P[state, state - 1] = 0.5

        # normalize the transition probabilities
        total_prob = np.sum(P[state, :])
        if total_prob > 0:
            P[state, :] /= total_prob

    durations = np.zeros((num_trials, num_states))
    for trial in range(num_trials):
        state_durations = np.zeros(num_states)
        current_state = np.random.randint(num_states)
        
        for _ in range(num_steps):
            next_state = np.random.choice(num_states, p=P[current_state, :])
            state_durations[current_state] += 1
            current_state = next_state
        
        durations[trial] = state_durations
    
    avg_durations = np.mean(durations, axis=0)

    return avg_durations