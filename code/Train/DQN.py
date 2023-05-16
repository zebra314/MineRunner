from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from tqdm import tqdm
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk
total_rewards = []

class replay_buffer():
    '''
    A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.

        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish
        
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.

        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''

    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 4  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.
        
        Parameter:
            states: a batch size of states
        
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
    
class Agent():
    def __init__(self):
        """
        The agent learning how to control the action of the cart pole.
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.epsilon = 0.2 # chance of taking a random action instead of the best
        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1", "jump 1"]
        self.n_actions = len(self.actions)  # the number of actions
        self.count = 0

        self.learning_rate = 0.0002
        self.gamma = 0.97
        self.batch_size = 32
        self.capacity = 10000

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network
        
        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.canvas = None
        self.root = None

    def learn(self):
        '''
        - Implement the learning function.
        - Here are the hints to implement.
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done this for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            None (Don't need to return anything)
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # Begin your code
        # TODO
        observations, actions, rewards, next_observations, done = self.buffer.sample(self.batch_size)
        
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_observations = torch.FloatTensor(np.array(next_observations))
        done = torch.BoolTensor(done)
        
        evaluate = self.evaluate_net(observations).gather(1, actions.reshape(self.batch_size, 1))
        nextMax = self.target_net(next_observations).detach()
        target = rewards.reshape(self.batch_size, 1) + self.gamma * nextMax.max(1)[0].view(self.batch_size, 1)\
                                                                  * (~done).reshape(self.batch_size, 1)
        
        MSE = nn.MSELoss()
        loss = MSE(evaluate, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # End your code
        torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")

    def act(self, world_state, agent_host, current_r):
        """
        Take one action in response to the current world state
        """
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # Most recent observation
        self.logger.debug(obs)

        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0

        current_s = (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % \
                          (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        
        # Choose an action
        # if random.random() < self.epsilon:

        #　bug太多 先以隨機的方式來選擇action
        # 修完之後再把下面else部份 uncomment (一樣有bug)
        # Random action
        self.logger.info("Random action")
        action_index = random.randint(0, self.n_actions - 1)

        # else:
            # Choose the best action based on the evaluate net
            # with torch.no_grad():
                # input = self.evaluate_net.forward(torch.FloatTensor(current_s))
                # action_index = torch.argmax(input).item()

        # Take the chosen action
        chosen_action = self.actions[action_index]
        agent_host.sendCommand(chosen_action)

        # Update the replay buffer
        if self.count > 0:
            self.buffer.insert(self.previous_observation, int(self.previous_action), 
                                current_r, obs, int(False))

        self.previous_observation = obs
        self.previous_action = action_index
        self.count += 1

        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        
        self.prev_s = None
        self.prev_a = None
        
        is_first_action = True
        
        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:

            current_r = 0
            
            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # Update the replay buffer and perform learning
        if self.count > 0:
            self.buffer.insert(self.previous_observation, 
                               self.previous_action, 
                               current_r, world_state, False)
            self.learn()

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r
    
        return total_reward
    
if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

agent = Agent()
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

# -- set up the mission -- #
mission_file = './tutorial_6.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

# add 20% holes for interest
for x in range(1,4):
    for z in range(1,13):
        if random.random()<0.1:
            my_mission.drawBlock( x,45,z,"lava")

max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 150

cumulative_rewards = []
for i in range(num_repeats):

    print()
    print('Repeat %d of %d' % ( i+1, num_repeats ))
    
    my_mission_record = MalmoPython.MissionRecordSpec()

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2.5)

    print("Waiting for the mission to start", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()

    # -- run the agent in the world -- #
    cumulative_reward = agent.run(agent_host)
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [ cumulative_reward ]

    # -- clean up -- #
    time.sleep(0.5) # (let the Mod reset)

print("Done.")

print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)
    