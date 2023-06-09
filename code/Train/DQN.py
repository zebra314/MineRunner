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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
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
    def __init__(self,  num_actions, hidden_layer_size=80):
        super(Net, self).__init__()
        self.input_state = 4  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer


    def forward(self, states):
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
        self.actions = ["move 1", "turn 0.5", "turn -0.5", "jump 1"]
        # self.actions = ["move  1", "jumpmove  1", "turn 1", "turn  -1"]
        # self.actions = ["moveForward 1", "set Yaw 45"]
        self.n_actions = len(self.actions)  # the number of actions
        self.count = 0

        # self.epsilon = 0.2 # chance of taking a random action instead of the best
        # self.epsilon = 0.95  # ???憪? epsilon ???
        # self.epsilon_decay_rate = 0.8 # epsilon ???銵唳?????
        # self.epsilon_min = 0.15  # epsilon ??????撠????
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.batch_size = 5
        self.capacity = 50000

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        # self.evaluate_net.load_state_dict(torch.load("../../asset/Tables/DQN_2023-06-06_07-45.pt"))
        self.target_net = Net(self.n_actions)  # the target network
        # self.target_net.load_state_dict(torch.load("../../asset/Tables/DQN_2023-06-06_07-45.pt"))
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
        self.action_state = []
        # self.yaw_bins = self.init_bins(0, 360, 8)
    
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
        # observations = observations.astype(float)
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
        

    def stopAction(self, agent_host, action_index):
        if action_index == None:
            return
        # jumpforward 1
        # elif action_index == 3:
        #     agent_host.sendCommand('move 0')
        #     agent_host.sendCommand('jump 0')
        #     return
        action = self.actions[action_index]
        action_substring = action.split(" ")
        stop_action = action_substring[0] + " 0"
        agent_host.sendCommand(stop_action)
        return
    
    def act(self, world_state, agent_host, prev_r, is_first_action):
        
        """
        Take one action in response to the current world state
        """
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # Most recent observation

        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_yaw = (float(obs[u'Yaw']) + 360) % 360
        # if current_yaw >= 315 or (current_yaw >= 0 and current_yaw < 45):
        #     yaw_discrete = 0
        # elif current_yaw >= 45 and current_yaw < 135:
        #     yaw_discrete = 1
        # elif current_yaw >= 135 and current_yaw < 225:
        #     yaw_discrete = 2
        # elif current_yaw >= 225 and current_yaw < 315:
        #     yaw_discrete = 3

        # yaw_discrete = int(((current_yaw + 45) % 360) // 90)   #整除成4等分
        # yaw_discrete = int(((current_yaw + 22.5) % 360) // 45)  #整除成8等分
        # yaw_discrete = int(((current_yaw + 22.5) % 360) // 22.5)   #整除成16等分

        current_XPos = float(obs[u'XPos'])
        current_ZPos = float(obs[u'ZPos'])
        current_YPos = int(obs[u'YPos'])
        # yaw_discretize = self.discretize_value(current_yaw, self.yaw_bins)
        # XPos_discrete = self.coordinate_discretize(current_XPos)
        # ZPos_discrete = self.coordinate_discretize(current_ZPos)
        current_state = (round(current_XPos, 1), round(current_YPos, 1), round(current_ZPos, 1), round(current_yaw, 1))
        # stop prev action after observation of current state
        self.stopAction(agent_host, self.prev_a)
        if not(world_state.is_mission_running) or bool(obs[u'IsAlive']) == False or int(obs[u'Life']) == 0:
            done = True
        else:
            done = False
        # Update the replay buffer
        if self.count > 0 and not is_first_action:
            self.buffer.insert(self.prev_s, self.prev_a, 
                                prev_r, current_state, int(done))

        self.count += 1
        # print(f'Current buffer size is: {len(self.buffer)}')
        if agent.count >= 50:
            # print(f'current buffer is: {self.buffer.memory}')
            agent.learn()
        # ----next action-----
        # Choose an action
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.count / self.EPS_DECAY)
            
        if random.uniform(0,1) < eps_threshold:
            self.logger.info("Explore")
            action_index = random.randint(0, self.n_actions - 1)
        else:
            # Choose the best action based on the evaluate net
            self.logger.info("Exploit")
            with torch.no_grad():
                action_index = torch.argmax(self.evaluate_net.forward(torch.FloatTensor(current_state))).item()
            
        chosen_action = self.actions[action_index]
        action_state_list = [chosen_action, current_state]
        self.action_state.append(tuple(action_state_list))
        self.logger.info("Taking q action: %s" % chosen_action)
        print(f'Current world state is:{current_state}, done is: {done}')
        # Take the chosen action
        try:
            agent_host.sendCommand(self.actions[action_index])
        except RuntimeError as e:
              self.logger.error("Failed to send command: %s" % e)
        self.prev_s = current_state
        self.prev_a = action_index
        # print update state information
        # If there are some new observations
        # while True and not done:
        #     latest_ws = agent_host.peekWorldState()
        #     if latest_ws.number_of_observations_since_last_state > 0:
        #         obs_text = latest_ws.observations[-1].text
        #         obs = json.loads(obs_text)
        #         next_yaw = (float(obs[u'Yaw']) + 360) % 360
        #         next_XPos = float(obs[u'XPos'])
        #         next_ZPos = float(obs[u'ZPos'])
        #         next_YPos = int(obs[u'YPos'])
        #         next_state = (round(next_XPos, 1), round(next_YPos, 1), round(next_ZPos, 1), round(next_yaw, 1))
        #         print(f'Next state is: {next_state}')
        #         break
        # self.stopAction(chosen_action, agent_host)
            
        # ??湔?? epsilon
        # self.epsilon *= self.epsilon_decay_rate
        # self.epsilon = max(self.epsilon, self.epsilon_min)

        # self.previous_observation = current_state
        # self.previous_action = action_index
        

        return
        
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
                        # # stop prev action after choose a new action
                        # self.stopAction(agent_host, self.prev_a)
                        self.act(world_state, agent_host, current_r, is_first_action)
                        total_reward += current_r
                        # print(f'Mission running is: {world_state.is_mission_running}')
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
                        print(f'current reward is:{current_r}')
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                        print(f'current reward is:{current_r}')
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        # # stop prev action after observation of current state
                        # self.stopAction(agent_host, self.prev_a)
                        self.act(world_state, agent_host, current_r, is_first_action)
                        total_reward += current_r
                        print(f'Total reward is: {total_reward}')
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


def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('Steve')
    plt.xlabel('epoch')
    plt.ylabel('rewards')


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
mission_file = './new_map_xml/tutorial_6.xml'
# mission_file = './new_map_xml/20230605_2.xml'
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
    num_repeats = 1000

cumulative_rewards = []
for i in range(num_repeats):
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
print(f'Action to state corresponding list:{agent.action_state}')
print("Done.")

print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)

os.makedirs("../../asset/Tables", exist_ok=True)
os.makedirs("../../asset/Rewards", exist_ok=True)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
np_file_path = f"../../asset/Rewards/DQN_rewards_{current_time}.npy"
np.save(np_file_path, np.array(cumulative_rewards))
DQN_file_path = f'../../asset/Tables/DQN_{current_time}.pt'
torch.save(agent.target_net.state_dict(), DQN_file_path)

initialize_plot()
plt.plot(cumulative_rewards)
plt.show()
plt.close()
exit(0)