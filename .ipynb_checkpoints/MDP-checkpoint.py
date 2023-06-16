"""
Parking Model MDP to solve Bandit Problem
"""
from sklearn import kernel_ridge
import numpy as np
import random

class BanditMDP:
    """
    Paramters:
        train_data - tuple of 2D array containing X and Y training data
               in their corresponding cluster
        train_data - tuple of array containing X and Y testing data
        max_bins - list of the total data points for each cluster
    """
    def __init__(self, train_data, test_data, max_bins):
        # Guassian Kernal Regression Model
        lambda_val = 0.1
        gamma = 0.001
        self.rrg = kernel_ridge.KernelRidge(alpha=lambda_val, kernel='rbf', gamma=gamma)\
        # Initialize max bins
        self.bins_size = len(max_bins)
        self.max_bins = max_bins
        # empty intital state
        self.state = BanditState([0]*self.bins_size)
        # initialize data
        self.X_train = train_data[0]
        self.Y_train = train_data[1]
        self.X_test = test_data[0]
        self.Y_test = test_data[1]
        # get possible operators based on bins
        self.operators = [0]
        self.action_to_index = {0:0}
        for i in range(1,self.bins_size+1):
            self.operators.append(i)
            self.action_to_index[i] = len(self.operators)-1
            self.operators.append(-i)
            self.action_to_index[-i] = len(self.operators)-1
        # Initialize Q-Table and Policy
        self.policy = {}
        self.q_table = {}
        self.q_table[self.state] = [0]*len(self.operators)
        # Goal state
        self.goal_samples = 0
        # MDP Parameters
        self.alpha = 1
        self.epsilon = 0.8
        self.rng = random.Random()
        
    """
    Get the training data corresponding to current state
    Return:
        x - training data input
        y - training data output
    """
    def get_data(self, curr_state):
        x = []
        y = []
        for i in range(self.bins_size):
            count = curr_state.bins[i]
            if count != 0:
                for j in range(count):
                    x.append(self.X_train[i][j])
                    y.append(self.Y_train[i][j])
                
        return np.array(x),np.array(y)

    """
    Preform a state transition based on a given action
    Parameter:
        action - given action to transition the current state
    Return:
        next_state - next state before the transition
    """
    def transition(self, action):
        return self.state.move(action, self.max_bins)

    """
    Run Guassian Kernel Ridge Regression to predict parking occupancy based on the current
    state and compute the mean absolute error(mae) using the testing data
    Parameter:
        curr_state - the state to be trained and tested
    Reutrn:
        'accuracy' - one minus mae
    """
    def reward(self, curr_state):
        x_train, y_train = self.get_data(curr_state)
        if x_train.size == 0:
            return 0
        self.rrg.fit(x_train, y_train)
        y_pred = self.rrg.predict(self.X_test)
        mae = np.mean(np.abs(self.Y_test-y_pred))
        return 1-mae

    """
    Set the amount of sample desired for the model
    Parameter:
        goal_samples - goal state total amount of samples 
    """
    def set_goal(self, goal_samples):
        self.goal_samples = goal_samples

    """
    Determine if the given state is the goal state
    Parameter:
        curr_state - state want to test if it is a goal state
    Return: True or False
    """
    def is_goal(self, curr_state):
        samples = sum(curr_state.bins)
        if samples == self.goal_samples:
            return True
        return False
        
    """
    Based on the q-table determine the optimal policy for each state
    """
    def get_policy(self):
        for key in self.q_table.keys():
            value = self.q_table[key]
            index = np.argmax(np.array(value))
            self.policy[key] = self.operators[index]

    """
    Compute one q-update based on the given action
    Parameter:
        action - given action to transition the current state
    """
    def q_update(self, action):
        next_state = self.transition(action)
        index = self.action_to_index[action]

        if next_state not in self.q_table.keys():
            self.q_table[next_state] = [0]*len(self.operators)
            
        if action != 0:
            sample = max(self.q_table[next_state]) - 0.01
            self.q_table[self.state][index] = (1-self.alpha)*self.q_table[self.state][index] + self.alpha*sample
        elif self.is_goal(self.state):
            # Don't repeat training and testing model if already done for that state
            if self.q_table[self.state][0] == 0:
                self.q_table[self.state][0]= self.reward(next_state)
            next_state = BanditState([0]*self.bins_size)

        self.state = next_state
        #print('new val:', self.q_table[self.state])

    """
    Get the next action, 1-epsilon probability of chosing the best action
    and epsilon probability of chosing a random action
    Return:
        action - next action based on the greedy exploration function
    """
    def get_action(self):
        coin = self.rng.random()
        if coin >= self.epsilon:
            index = np.argmax(np.array(self.q_table[self.state]))
            return self.operators[index]
        else:
            return self.rng.choice(self.operators)

    """
    Run one iteration of q-learning algorithm and update q-table accordingly
    """
    def run_iteration(self):
        action = self.get_action()
        if self.state.can_move(action, self.max_bins):
            #print('curr_state',self.state.bins)
            #print('action:',action)
            self.q_update(action)

    """
    Get the optimal state based on the policy computed from the q-table and print
    out the path taken to the goal state
    """
    def get_solution(self):
        state_list = set()
        self.get_policy()
        curr_state = BanditState([0]*self.bins_size)
        state_list.add(curr_state)
        while self.policy[curr_state] != 0:
            print('curr_state',curr_state.bins)
            action = self.policy[curr_state]
            print('action:',action)
            curr_state = curr_state.move(action, self.max_bins)
            if curr_state in state_list:
                break
            state_list.add(curr_state)
        print('Solution State', curr_state.bins)
        print('Accuracy', self.reward(curr_state))
        return curr_state.bins
        
class BanditState:
    """
    State represent the current Parking-Model
    Parameter:
        bins - an array with count of the number of sample for each clusters
    """
    def __init__(self, bins):
        self.bins = bins

    def copy(self):
        rep = self.bins.copy()
        return BanditState(rep)

    """
    Transition state to another given an action
    Parameters:
        action - valid operators
        max_bins - max count of sample in each bin
    Return:
        next state based on given action
    """
    def move(self, action, max_bins):
        next_state = self.copy()
        if self.can_move(action, max_bins) and action != 0:
            index = abs(action) - 1
            if action < 0:
                next_state.bins[index] -= 1
            elif action > 0:
               next_state.bins[index] += 1

        return next_state

    """
    Check if the current state can take the given action
    Parameters:
        action - given action to take
        max_bins - array with max count of sample for each clusters
    Return: True or False
    """
    def can_move(self, action, max_bins):
        index = abs(action) - 1
        if action < 0 and self.bins[index] == 0:
            return False
        elif action > 0 and self.bins[index] == max_bins[index]:
            return False
        return True

    """
    Check if given state is equal to current state
    Parameter:
        other - given state to compare with current state
    Return: True or False
    """
    def __eq__(self, other):
        for i in range(len(self.bins)):
            if self.bins[i] != other.bins[i]:
                return False
        return True

    """
    Compute hash of the current state
    """
    def __hash__(self):
        return hash((tuple(self.bins)))

