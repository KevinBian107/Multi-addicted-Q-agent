'''Major model used in this study'''

import numpy as np
import random

class Addicted_Q_Agent:
    def __init__(self,
                 alpha: float = 0.1,
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
        Addicted Q Agent class
        
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
        
        methods:
            - learning
            - resimulate average vissits with best Q matrix after learning
            - random walk of the agents
        '''
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_trials = num_trials
        self.num_states = num_states
        self.num_actions = num_actions
        self.initial_dopamine_surge = initial_dopamine_surge
        self.dopamine_decay_rate = dopamine_decay_rate
        self.reward_states = reward_states
        self.drug_reward = drug_reward
        self.addicted_agent = addicted_agent
        
        # initialize Q (action values) table, errors, and state_action values
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.rpe = np.zeros((self.num_trials, self.num_states, self.num_actions))
        self.Q_across_trials = np.zeros((self.num_trials, self.num_states, self.num_actions))

        assert len(self.reward_states) <= self.num_states, "reward states must be less than number of states"


    def learning(self, greedy=False):
        '''
        Simulating addicted Q learning agent, using epsilon greedy policy

        args:
            - greedy: boolean argument for only exploration

        return:
            - Reward prediction error over trials
            - Q table over trials
        '''

        for trial in range(self.num_trials):
            dopamine_surge = self.initial_dopamine_surge * (self.dopamine_decay_rate ** trial)

            for state in range(self.num_states):
                action = np.argmax(self.Q[state])
                if not greedy:
                    if random.uniform(0, 1) < self.epsilon:
                        action = random.randint(0, self.num_actions - 1)

                if action == 0:
                    next_state = min(state + 1, self.num_states - 1)
                elif action == 1:
                    next_state = max(state - 1, 0)
                else:
                    next_state = state
                
                # reward activation
                reward = self.drug_reward if next_state in [reward_state for reward_state in self.reward_states] else 0

                # update TD
                max_next_Q = np.max(self.Q[next_state])

                
                delta = reward + self.gamma * max_next_Q - self.Q[state, action]

                if self.addicted_agent:
                    delta = max(reward + self.gamma * max_next_Q - self.Q[state, action] + dopamine_surge * reward, dopamine_surge * reward)
                
                # update table
                self.Q[state, action] = self.Q[state, action] + self.alpha * delta
                self.rpe[trial, state, action] = delta
                self.Q_across_trials[trial, state, action] = self.Q[state, action]

        return self.rpe, self.Q_across_trials


    def resimulate_state_durations(self, num_re_trials=100, max_action_per_trial=10):
        '''
        Re-simulate average state durations using Q-values and calculate the average duration in each state.
        In stochastic processes, average visits is a key metrics of the chain or the  process, so we try to do the same
        simulation wise here with using best learned Q
        
        args:
            - num_trials: number of trials
        
        return:
            - average duration in each state
        '''

        assert self.Q_across_trials.sum().sum() != 0, 'need to fit for learning first'

        Q = self.Q_across_trials[-1]
        state_durations = np.zeros((num_re_trials, self.num_states))

        for trial in range(num_re_trials):
            state = np.random.randint(self.num_states)
            
            counter = 0
            while True:
                action = np.argmax(Q[state])
                print(action)
                if action == 0 and state < self.num_states - 1:
                    next_state = state + 1
                elif action == 1 and state > 0:
                    next_state = state - 1
                else:
                    next_state = state
                
                state_durations[trial, state] += 1
                state = next_state

                counter+=1
                
                # stop condition
                if state == self.num_states - 1:
                    break

                if counter == max_action_per_trial:
                    break
        
        avg_durations = np.mean(state_durations, axis=0)

        return avg_durations


    def random_walk(self, num_trials=100, num_steps=1000):
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
        
        P = np.zeros((self.num_states, self.num_states))
        
        # transition probability matrix
        for state in range(self.num_states):
            if state < self.num_states - 1:
                P[state, state + 1] = 0.5
            if state > 0:
                P[state, state - 1] = 0.5

            # normalize the transition probabilities
            total_prob = np.sum(P[state, :])
            if total_prob > 0:
                P[state, :] /= total_prob

        durations = np.zeros((num_trials, self.num_states))
        for trial in range(num_trials):
            state_durations = np.zeros(self.num_states)
            current_state = np.random.randint(self.num_states)
            
            for _ in range(num_steps):
                next_state = np.random.choice(self.num_states, p=P[current_state, :])
                state_durations[current_state] += 1
                current_state = next_state
            
            durations[trial] = state_durations
        
        avg_durations = np.mean(durations, axis=0)

        return avg_durations