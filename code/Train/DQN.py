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
    
    def __len__(self):
        return len(self.memory)

class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''

    def __init__(self,  num_actions, hidden_layer_size=100):
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
        
        # self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        # self.actions = ["move 1","move 0.5","move 0.5", "move -1", "turn 0.5", "turn -0.5", "jump 1"]
        self.actions = ["moveForward 1", "set Yaw 45"]
        self.n_actions = len(self.actions)  # the number of actions
        self.count = 0

        # self.epsilon = 0.2 # chance of taking a random action instead of the best
        self.epsilon = 1.0  # 初始 epsilon 值
        self.epsilon_decay_rate = 0.9999 # epsilon 的衰減率
        self.epsilon_min = 0.15  # epsilon 的最小值

        self.learning_rate = 0.001
        self.gamma = 0.9
        self.batch_size = 32
        self.capacity = 50000

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
        self.yaw_bins = self.init_bins(0, 360, 8)
    def init_bins(self, lower_bound, upper_bound, num_bins):
        # Begin your code
        """
        Explain code:
        linspace can part [lower_bound, upper_bound] into {num} evenly spaced points
        To slice interval into {num_bins} subinterval, we need {num_bins+1} points
        return np array that excluding the first and last element
        """
        return np.linspace(lower_bound, upper_bound, num = num_bins+1)[1:-1]
        # End your code
    def discretize_value(self, value, bins):
        # Begin your code
        """
        Explain code:
        np.digitize let 2 neighbor points in bins is considered as a interval
        ex: bins has 4 points, so it has 3 interval
        return value is interval index which given value is located at including lower_bound and upper_bound in init_bins
        if in the first interval, return value is 0, and so on
        """
        return np.digitize(value, bins)
        # End your code
    def addTermOfXZ(self, yaw_interval):
        x_add_term = 0
        z_add_term = 0
        if yaw_interval == 0:
            x_add_term = 0
            z_add_term = 1
        elif yaw_interval == 1:
            x_add_term = -0.70711
            z_add_term = 0.70711
        elif yaw_interval == 2:
            x_add_term = -1
            z_add_term = 0
        elif yaw_interval == 3:
            x_add_term = -0.70711
            z_add_term = -0.70711
        elif yaw_interval == 4:
            x_add_term = 0
            z_add_term = -1
        elif yaw_interval == 5:
            x_add_term = 0.70711
            z_add_term = -0.70711
        elif yaw_interval == 6:
            x_add_term = 1
            z_add_term = 0
        elif yaw_interval == 7:
            x_add_term = 0.70711
            z_add_term = 0.70711
        return x_add_term, z_add_term
    def moveStraight(self, agent_host, factor, world_state):
        flag = False
        move_speed = factor * 0.5
        # agent_host.sendCommand('move {}'.format(move_speed))
        done = False
        while not done and flag is False:
            latest_ws = agent_host.peekWorldState()
            print(f'Move straight, Latest world state is: {latest_ws}')
            # If there are some new observations
            if latest_ws.number_of_observations_since_last_state > 0:
                obs_text = latest_ws.observations[-1].text
                obs = json.loads(obs_text)
                print(f'Peek World State is:{obs}')
                current_ZPos = float(obs[u'ZPos'])
                current_XPos = float(obs[u'XPos'])
                current_yaw = float(obs[u'Yaw'])
                current_yaw_interval = self.discretize_value(current_yaw, self.yaw_bins)
                x_add_term, z_add_term = self.addTermOfXZ(current_yaw_interval)
                print(f'Current yaw is: {current_yaw_interval}, x term is{x_add_term}, z term is {z_add_term}')
                target_ZPos = current_ZPos + z_add_term
                target_XPos = current_XPos + x_add_term
                # use manhattan distance to calculate distance between current and target
                # manhattan distance: x + z 
                # 1 gaussian distance ~ 1.414 manhattan distance
                # target_manhattan = current_ZPos + current_XPos + 1.414
                print(f'Init Current XPos is {current_XPos}, ZPos is {current_ZPos}, target XPos is {target_XPos}, target ZPos is {target_ZPos}')
                while not done and abs(current_XPos - target_XPos) + abs(current_ZPos - target_ZPos) > 0.5:
                    time.sleep(0.1)
                    agent_host.sendCommand('move {}'.format(move_speed))
                    latest_ws = agent_host.peekWorldState()
                    # If there are some new observations
                    if latest_ws.number_of_observations_since_last_state > 0:
                        obs_text = latest_ws.observations[-1].text
                        obs = json.loads(obs_text)
                        # print(f'Peek World State is:{obs}')
                        current_ZPos = float(obs[u'ZPos'])
                        current_XPos = float(obs[u'XPos'])
                        if latest_ws.is_mission_running == False or obs[u'IsAlive'] == False or int(obs[u'Life']) == 0:
                            done = True
                        else:
                            done = False
                        agent_host.sendCommand('move {}'.format(move_speed))
                        print(obs)
                        print(f'Current XPos is {current_XPos}, ZPos is {current_ZPos}, target XPos is {target_XPos}, target ZPos is {target_ZPos}')
                flag = True
                agent_host.sendCommand('move 0')
        print(f'move straight {factor} success!')
    # forward is look at Z position
    def turnDegree(self, agent_host, factor, world_state):
        # obs_text = world_state.observations[-1].text
        # obs = json.loads(obs_text)
        # print(f'Argument World State is:{obs}')
        flag = False
        turn_speed = factor * 0.5
        agent_host.sendCommand('turn {}'.format(turn_speed))
        isAlive = True
        last_timeAlive = None
        while isAlive and isAlive and flag is False:
            latest_ws = agent_host.peekWorldState()
            print(f'TurnDegree, Latest world state is: {latest_ws}')
            # If there are some new observations
            if latest_ws.number_of_observations_since_last_state > 0:
                obs_text = latest_ws.observations[-1].text
                obs = json.loads(obs_text)
                print(f'Peek World State is:{obs}')
                current_yaw = int(obs[u'Yaw'])
                # factor = 1: turn west, factor = -1: turn east
                init_target_yaw = factor * 45 + current_yaw
                if factor == 1:
                    target_yaw = (init_target_yaw) % 360
                else:
                    target_yaw = init_target_yaw + 360 if init_target_yaw < -360 else init_target_yaw
                print(f'Init Current yaw is {current_yaw}, target yaw is {target_yaw}')
                while isAlive and abs(current_yaw - target_yaw) > 5:
                    time.sleep(0.1)
                    agent_host.sendCommand('turn {}'.format(turn_speed))
                    latest_ws = agent_host.peekWorldState()
                    # If there are some new observations
                    if latest_ws.number_of_observations_since_last_state > 0:
                        obs_text = latest_ws.observations[-1].text
                        obs = json.loads(obs_text)
                        current_yaw = int(obs[u'Yaw'])
                        timeAlive = obs[u'TimeAlive']
                        if timeAlive == last_timeAlive:
                            isAlive = False
                        last_timeAlive = timeAlive
                        agent_host.sendCommand('turn {}'.format(turn_speed))
                        print(obs)
                        print(f'Current yaw is {current_yaw}, target yaw is {target_yaw}, isAlive is: {isAlive}')
                flag = True
                agent_host.sendCommand('turn 0')
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
        
        # Forward the data to the evaluate net and the target net.
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_observations = torch.FloatTensor(np.array(next_observations))
        done = torch.BoolTensor(done)
        # print(f'Observation is: {observations}')
        # Compute the loss
        evaluate = self.evaluate_net(observations).gather(1, actions.reshape(self.batch_size, 1))
        nextMax = self.target_net(next_observations).detach()
        target = rewards.reshape(self.batch_size, 1) + self.gamma * nextMax.max(1)[0].view(self.batch_size, 1)\
                                                                  * (~done).reshape(self.batch_size, 1)
        # Zero-out the gradients
        MSE = nn.MSELoss()
        loss = MSE(evaluate, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # End your code
        torch.save(self.target_net.state_dict(), "DQN.pt")

    def act(self, world_state, agent_host, current_r):
        """
        Take one action in response to the current world state
        """
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # Most recent observation

        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_yaw = (float(obs[u'Yaw']) + 360) % 360
        current_XPos = int(obs[u'XPos'])
        current_YPos = int(obs[u'YPos'])
        current_ZPos = int(obs[u'ZPos'])
        print(f'Current yaw is: {current_yaw}')
        current_s = (current_XPos, current_YPos, current_ZPos, self.discretize_value(current_yaw, self.yaw_bins))
        print(f'Current state is: {current_s}')
        if world_state.is_mission_running or obs[u'IsAlive'] == False or int(obs[u'Life']) == 0:
            done = True
        else:
            done = False

        # Choose an action
        if random.uniform(0,1) < self.epsilon:
            self.logger.info("Explore")
            action_index = random.randint(0, self.n_actions - 1)
        else:
            # Choose the best action based on the evaluate net
            self.logger.info("Exploit")
            with torch.no_grad():
                action_index = torch.argmax(self.evaluate_net.forward(torch.FloatTensor(current_s))).item()
            
        # chosen_action = self.actions[action_index]
        self.logger.info("Taking q action: %s" % self.actions[action_index])
        # Take the chosen action
        try:
          # if world_state.is_mission_running:
          #   print('Mission is still running')
          # else:
          #   print('Mission is not running')
          # # agent_host.sendCommand('move 0')
          # agent_host.sendCommand('move 1')
          # if world_state.is_mission_running:
          #   print('Mission is still running')
          # else:
          #   print('Mission is not running')
            # move forward
            if action_index == 0:
                # agent_host.sendCommand("strafe 1")
                self.turnDegree(agent_host, 1, world_state)
                
                # agent_host.sendCommand('move 1')
                time.sleep(1)
                # world_state = agent_host.getWorldState()
                # obs_text = world_state.observations[-1].text
                # obs = json.loads(obs_text)
                # print(f'After move forward, World State is:{obs}')
            # move backward
            # elif a == 1:
            #     # agent_host.sendCommand("strafe -1")
            #     # self.moveStraight(agent_host, -1, world_state)
            #     agent_host.sendCommand('move -1')
            #     time.sleep(1)
            # elif a == 1:
            #     agent_host.sendCommand("move 1")
            #     agent_host.sendCommand("jump 1")
            #     time.sleep(1)
            else:
                self.moveStraight(agent_host, 1, world_state)
                # agent_host.sendCommand('move 1')
                time.sleep(1)
                # agent_host.sendCommand("turn 45")
            # elif a == 4:
            #     self.turnDegree(agent_host, -1, world_state)
            #     agent_host.sendCommand("turn -45")
            
            # agent_host.sendCommand(self.actions[a])
            self.prev_s = current_s
            self.prev_a = 0

        except RuntimeError as e:
              self.logger.error("Failed to send command: %s" % e)
        
        # time.sleep(1)
        # Update the replay buffer
        if self.count > 0:
            self.buffer.insert(self.previous_observation, int(self.previous_action), 
                                current_r, current_s, int(done))

        self.count += 1
        # print(f'Current buffer size is: {len(self.buffer)}')
        if agent.count >= 50:
            # print(f'current buffer is: {self.buffer.memory}')
            agent.learn()
            
        # 更新 epsilon
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_min)

        self.previous_observation = current_s
        self.previous_action = action_index
        

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

# # add 20% holes for interest
# for x in range(1,4):
#     for z in range(1,13):
#         if random.random()<0.1:
#             my_mission.drawBlock( x,45,z,"lava")

max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 5000

cumulative_rewards = []
for i in range(num_repeats):
    print(f'yaw_bins is:{agent.yaw_bins}')
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

os.makedirs("../Rewards", exist_ok=True)
np.save("../Rewards/DQN_rewards.npy", np.array(cumulative_rewards))
    
