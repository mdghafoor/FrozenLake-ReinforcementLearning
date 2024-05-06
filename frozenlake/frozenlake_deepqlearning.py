# -*- coding: utf-8 -*-
__author__ = 'Muhammad Ghafoor'
__version__ = '1.0.1'
__email__ = "mdhgafoor@outlook.com"

"""
File Name: frozenlake_deepqlearning.py
Description: Deep Q Reinforcement learning project using OpenAI's Gymnasium and 
             inspiration from course guidelines by Andrew Ng. 
             https://www.coursera.org/specializations/machine-learning-introduction
             https://gymnasium.farama.org   

             This script executes the following Deep Q learning with Experience Replay algorithm:
                1. Initialize memory buffer D with capacity N
                2. Initialize Q network with random weights w
                3. Initialize target Q^ network with random weights w- = w
                4. For episode i=1 to M do:
                5      Receive initial observation state S_1
                6.     While not solved, or not fallen through environment do:
                7.         Observe state S_t and choose action A_t using ε greedy policy.
                8.         Take action A_t and recieve reward, and nexet state S_t+1
                9.         Store experience replay (S_t,A_t,R_t,S_t+1) in memory buffer.
                10.        Every C steps, perform update:
                11.            Sample random mini batch of experience replays from memory buffer D (S_j, A_j, R_j, S_j+1)
                12.            Set y_j to R_j if episodes terminate at j+1 else, set y_j to R_J + γ max_a' Q^(s_j+1,a')
                13.            Perform a gradient decent step on (y_j-Q(s_j,a_j;w))^2 with respect to Q Network weights w
                14.            Update the weights of target Q^ Network using soft update method
                15.        Every X steps, perform hard update:
                16.            Update weights of target Q^ Network with Q Network weights using hard upate method
                17.    end
                18. end 
            
             Important notes/explanations for algorithm: 
                1. Estimate Action Value function interatively using Bellman equation:
                    Q_(i+1) (s,a) = R + γ max_a' Q^_i (s',a')
                    This iterative method converges Q*(s,a) as i->infinity where Q*(s,a) is the optimal action-value function.
                    Using a neural network an estimate of Q(s,a) can be obtained where Q(s,a) approximately equals Q*(s,a) by adjustings
                    the weights of Q(s,a) at each iteration to minimize the MSE in the bellman equation. 
                2. In order to obtain the MSE to update Q(s,a), first the target values are required. These are obtained using the following formula:
                    y = R + γ max_a' Q(s',a';w) where w are the weights of the Q Network
                3. We adjust the weights w at each iteration by minimizing the following error:
                    R + γ max_a' Q(s',a';w) - Q(s,a;w)
                4. Since y_targets change at each iteration, in order to reduce oscillations and instability, a seperate neural network,
                    target Q-Network is created to generate y_targets. Therefore the formula is adjusted to:
                    R + γ max_a' Q^(s',a';w-) - Q(s,a;w) where w- is the weights of the target Q-Network and w are the weights of the Q-Network
                5. When updating the weights of the target Q-network using the weights of the Q-Network, a soft update approach is used every C steps.
                    This is controlled by hyperparameter TAU where:
                    w- <- TAU*w + (1-TAU)*w-.
                    This helps ensure y_targets update slowly to improve stability.
                    However, due to scarce rewards, a hard update is also performed every X steps where
                    w- <- w                 
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

class FrozenLake:
    """
    Deep Q learning agent for Frozen Lake environment.
    """

    #Set Hyperparameters used in learning
    MEMORY_SIZE = 1000 #Size of Memory Buffer
    GAMMA = 0.95 #Discount Factory
    ALPHA = 1e-3 #Learning Rate
    MINIBATCH_SIZE=32 #Batch size of memory entries used for training
    E_MIN = 1e-3 #Minimum ε value for epilson greedy policy
    E_DECAY = 0.994 #Rate of ε decay 
    TAU = 1e-3 #Soft update Parameter


    def __init__(self):
        """
        Initializes class and sets random seed for reproducibility.
        """
        tf.random.set_seed(0)


    def _environment_setup(self, map_name='4x4', render=None):
        """
        Creates desired Frozen Lake environment and obtains standard parameters of state size and number of actions.
        Args: 
            map_name (str, optional): Selects map name from "4x4" or "8x8". Default to "4x4"
            render: [str] Rendering mode. Options include 'human' for visual display, 'rgb-array' for image data, or None for no rendering.
        """
        self.env = gym.make('FrozenLake-v1', desc=None, map_name=map_name, is_slippery=False, render_mode=render if render is not None else None)
        self.state_size = (self.env.observation_space.n,)
        self.num_actions = self.env.action_space.n 


    def _define_networks(self):
        """
        Defines Q and Target Q neural networks. 
        """
        self.q_network = Sequential([
            Input(shape=self.state_size),
            Dense(units=16,activation='relu'),
            Dense(units=self.num_actions,activation='linear'),
        ])
        
        self.target_q_network = Sequential([
        Input(shape=self.state_size),
        Dense(units=16,activation='relu'),
        Dense(units=self.num_actions,activation='linear'),
        ])

        self.optimizer = Adam(learning_rate=self.ALPHA)
    

    def _update_target_network(self):
        """
        Updates the weights of a the target Q network using a soft update controled by hyperparameter Tau.
        Update Formula:
            w_target = (TAU * w) + (1 - TAU) * w_target
        where: 
            - w_target = weights of target q network
            - TAU = soft update parameter
            - w = weights of q network
        """
        for target_weights, q_net_weights in zip(self.target_q_network.weights, self.q_network.weights):
            target_weights.assign(self.TAU * q_net_weights + (1 - self.TAU) * target_weights)


    def _compute_loss(self, experiences):
        """
        Computes Mean Square Error loss of y_targets and q_values calculated. 
        Takes minibatch of experiences tuples to unpack its respective tensors, states, actions, rewards, next_states, and done.
        
        y_targets = { Rj                    if episodes termiante at steps j+1
                    { Rj + γ max Q^(s,a)    otherwise
           
        Args:
            experiences: [tuple] tuple of [states, actions, rewards, next_states, done]
        
        Returns:
            loss: [(TensorFlow Tensor(shape=(0,), dtype=int32))] MSE between y_targets and Q(s,a) values. 
        """
        #Unpack experience tuple
        states, actions, rewards, next_states, done = experiences 

        #Calculate max Q^(s,a)
        max_qsa = tf.reduce_max(self.target_q_network(next_states), axis=1)

        #Calculate y tagets based on formula in docstring
        y_targets = rewards+self.GAMMA*max_qsa*(1-done)

        #Get q values and reshape to match y targets
        q_values = self.q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))

        #Calculate loss
        loss = MSE(y_targets,q_values)

        return loss 
    

    @tf.function
    def _agent_learn(self, experiences):
        """
        Updates the weights of Q Network and Target Q^ Network by calculating loss, obtaining gradients, and applying optimization.
        Update occurs in seperate method using soft update approach controlled by hyperparamete TAU.

        Args:
            experiences: [tuple] tuple of [states, actions, rewards, next_states, done]
        """
        with tf.GradientTape() as tape:
            loss = self._compute_loss(experiences)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        self._update_target_network()
    

    def _get_action(self, q_values, epsilon=0.0):
        """
        Returns action to execute based on an ε greedy policy. 
            - With probability ε, an action chosen at random.
            - With probability (1 - ε), the action that yields the maximum Q value in q_values will be chosen. 

        Args:
        
            - q_values: [tf.Tensor] Q values returned by Q-Network. In Frozen Lake environment, 
                        this Tensor has shape [1,4] with dtype=tf.float32
            - epsilon: [float] Current value of epilson.

        Returns:
            action: [numpy.int64] Action chosen based on policy described above.
        """
        if random.random() > epsilon:
            return np.argmax(q_values.numpy()[0])
        else:
            return random.choice(np.arange(4))
    

    def _check_update_conditions(self, rewards_per_episode):
        """
        Checks if agent is ready for an update. Initial update occurs after a reward is obtained since 
        rewards are scarce in Frozen Lake environment. 

        Args:
            - rewards_per_episode: [list] List of rewards obtained in each episode
        
        Return:
            - update_condition_met: [bool] Boolean that lets agent know if update conditions are met.
        """
        if sum(rewards_per_episode)>0 and len(self.memory_buffer) > self.MINIBATCH_SIZE:
            return True
        else:
            return False 
    

    def _get_experiences(self):
        """
        Obtains random set of experiences of size determined by MINIBATCH_SIZE.
        States and Next_State parameters are converted into one-hot format for use by neural network.

        Returns:
            experiences tuple: [tuple] tuple containing random set of states, actions, rewards, next_states, and terminated values. 
        """
        experiences = random.sample(self.memory_buffer, k=self.MINIBATCH_SIZE)
        states = tf.one_hot(tf.convert_to_tensor(np.array([e.state for e in experiences]), dtype=tf.int32), depth=self.state_size[0])
        actions = tf.convert_to_tensor(np.array([e.action for e in experiences]), dtype=tf.int32)
        rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences]), dtype=tf.float32)
        next_states = tf.one_hot(tf.convert_to_tensor(np.array([e.next_state for e in experiences]), dtype=tf.int32), depth=self.state_size[0])
        terminated = tf.convert_to_tensor(np.array([e.done for e in experiences]).astype(np.uint8), dtype=tf.float32)

        return (states, actions, rewards, next_states, terminated)


    def _get_new_eps(self, epsilon):
        """
        Updates ε based on ε greedy policy
        """
        return max(self.E_MIN,self.E_DECAY*epsilon)


    def _save_model(self, model_filedir):
        """"
        Stores the trained agent in .keras format.

        Args:
            - model_filename: [string] file directory where to save model. 
        """
        self.q_network.save(model_filedir)
    

    def _create_figures(self, num_episodes, rewards_per_episode, epsilon_history, plot_save_filepath):
        """
        Creates plot to:
            1. Display sum of rewards over last 100 episodes for each episode
            2. Display decay of ε via the ε greedy policy. 

        Args:
            - num_episodes: [int] Number of total episodes
            - rewards_per_episode: [list] List containing rewards over all episodes
            - epsilon_history: [list] List recording changes in ε over all episodes
        """
        #Calculate sum of rewards for the last 100 episodes over all episodes
        sum_rewards = np.zeros(num_episodes)
        for x in range(num_episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        
        #Plot figures defined above
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle("Frozen Lake Deep-Q Learning Network")
        
        ax1.plot(sum_rewards)
        ax1.set_xlabel('Episodes', size=8)
        ax1.set_ylabel('Sum of Rewards', size=8)
        ax1.set_title('Sum of Rewards per last 100 Episodes', size=8)
        ax1.set_xticks(np.arange(0, num_episodes+1, 1000))
        ax1.set_yticks(np.arange(0, max(sum_rewards)+1, 10))

        ax2.plot(epsilon_history)
        ax2.set_xlabel('Episodes', size=8)
        ax2.set_ylabel('Epsilon', size=8)
        ax2.set_title('Epsilon value over all Episodes', size=8)
        ax2.set_xticks(np.arange(0, num_episodes+1, 1000))
        ax2.set_yticks(np.arange(0, max(epsilon_history)+0.01, 0.1))

        plt.tight_layout()

        fig.savefig(plot_save_filepath)


    def train_agent(self, model_filepath, map_name, plot_save_filepath, render=None, num_episodes=2000):
        """
        Trains Frozen Lake solving agent via Deep-Q Learning with Experience Replay.
        Exact algorithm description defined in file description.

        Args:
            - model_filepath: [str] Location for where to save model after training.
            - num_episodes: [int] Number of episodes to train over. 
        """
        #Set up environment and define nueral networks.
        self._environment_setup(map_name, render)
        self._define_networks()

        #Define size of memory buffer and format of experience tuple.
        self.memory_buffer = deque(maxlen=self.MEMORY_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        #Define hard update rate and set initial step value.
        self.hard_update_rate = 20
        step_count = 0

        #Set initial weights of network.
        self.target_q_network.set_weights(self.q_network.get_weights())
        
        #Initilize parameters to store history of rewards and epsilon.
        rewards_per_episode = np.zeros(num_episodes)
        epsilon = 1.0
        epsilon_history = [epsilon]
        
        #Begin Training over number of episodes defined
        for i in range(num_episodes):
            #Reset environment every episode
            self.state = self.env.reset()[0]
            terminated = False
            truncated = False

            #While current status is not terminated or truncated (did not fall through hole in frozen lake, or solve environment):
            while (not terminated and not truncated):
                #Obtain current state and q values assocaited with current state.
                state_one_hot = tf.one_hot(self.state, depth=self.env.observation_space.n)
                state_one_hot = tf.expand_dims(state_one_hot, axis=0)
                q_values = self.q_network(state_one_hot)
                
                #Obtain aciton based on q values and current epsilon.
                action = self._get_action(q_values,epsilon)

                #Execute action obtained above and record experiences
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.memory_buffer.append(self.experience(self.state,action,reward,next_state,terminated))
                self.state = next_state
                step_count += 1

            #If enviornemnt is solved, record reward
            if reward == 1:
                rewards_per_episode[i] = 1

            #Update agent if parameters are met.
            if self._check_update_conditions(rewards_per_episode):
                experiences = self._get_experiences()
                self._agent_learn(experiences)

            #Execute hard upate based on hard_update_rate
            if step_count > self.hard_update_rate:
                self.target_q_network.set_weights(self.q_network.get_weights())
                step_count = 0
            
            #Obtain new epislon based on epsilon greedy policy and record epsilon change
            epsilon = self._get_new_eps(epsilon)
            epsilon_history.append(epsilon)
            
            #Simple display of agent training based on sum of rewards results every 100 episodes
            if i%100==0:
                print(f'Episode: {i}, Epsilon: {epsilon}, Rewards over all episodes: {sum(rewards_per_episode)}')
    
        #Create figures based on performance observed
        self._create_figures(num_episodes, rewards_per_episode, epsilon_history, plot_save_filepath)

        #Save agent
        self._save_model(model_filepath)
    
    
    def test_agent(self, episodes, map_name="4x4", model_filepath=None, is_slippery=False, render=None):
        """
        Test model trained above over number of episodes determined by visually observing performance and recording success rate.

        Args:
            - epsiodes: [int] Number of episodes to view trained agent execute.
            - map_name: [str] Map name to test on. Options include "4x4" and "8x8".
            - model_filepath: [str] Location of saved model if not using same model just trained.
            - is_slippery: [bool] Optionally test model against more difficult slippery conditions.
        """
        #Load model if filepath provided
        if model_filepath:
            self.q_network = load_model(model_filepath)
        
        #Create environment
        self.env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode=render if render is not None else None)
        rewards = 0

        #Run environment and execute actions with trained agent.
        for i in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            truncated = False
            
            while (not terminated and not truncated):
                state_one_hot = tf.one_hot(state, depth = self.env.observation_space.n)
                state_one_hot = tf.expand_dims(state_one_hot, axis=0)
                q_values = self.q_network(state_one_hot)
                action = self._get_action(q_values)
                next_state, reward, terminated, truncated, _ = self.env.step(action) 
                state = next_state
                rewards += reward
        
        #Calculate and report success rate over all episodes
        success_rate = rewards/episodes*100
        print(f"This model produces a {success_rate}% success rate!")

    
if __name__=="__main__":
    """
    Example of how to run program above
    """
    model_filepath = 'FrozenLake-ReinforcementLearning/frozenlake/FrozenLakeDeepQLearning.keras'
    plot_save_filepath = 'FrozenLake-ReinforcementLearning/frozenlake/FrozenLakeDeepQLearningStats.png'
    map_name = "4x4"
    render = 'human'

    fl = FrozenLake()
    fl.train_agent(model_filepath, map_name, plot_save_filepath, num_episodes=2000)
    fl.test_agent(500, map_name, model_filepath, render=render)