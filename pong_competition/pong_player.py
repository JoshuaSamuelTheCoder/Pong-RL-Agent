import random
import itertools
import math
import gym
import numpy as np
import torch
from collections import namedtuple
import torch.optim as optim

import torch.nn.functional as F
from pong_env import PongEnv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MyModelClass(torch.nn.Module):
    
    def __init__(self):
        super(MyModelClass, self).__init__()
        self.linear1 = torch.nn.Linear(7, 5)
        self.linear2 = torch.nn.Linear(5, 3)
        self.steps = 0
        
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
        


class PongPlayer(object):

    def __init__(self, save_path, load=False):
        self.build_model()
        self.build_optimizer()
        self.steps = 0
        self.save_path = save_path
        if load:
            self.load()

    def build_model(self):

        self.model = MyModelClass()

    def build_optimizer(self):

        self.dqn = MyModelClass()
        self.optimizer = torch.optim.RMSprop(self.dqn.parameters(), lr=0.0001)
        

    def get_action(self, state):
        self.steps += 1
        choice = random.random()
        eps_treshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * self.steps / EPS_DECAY)
        if choice > eps_treshold:
            with torch.no_grad():
                tensor = self.model(torch.tensor(state, dtype=torch.float32))
                out = tensor.max(0)[1].numpy()
                return out
        else:
            out =  torch.tensor([[random.randrange(2)]],device = device , dtype=torch.long).numpy()[0, 0]
            return out
        
    def train(self):
        if len(memory.memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
        
        listOfT1 = []
        for s in batch.next_state:
            listOfT1.append(torch.tensor(s, dtype = torch.float32).view(1, -1))
            
        non_final_next_states = torch.cat([s for s in listOfT1 if s is not None])
    
        listOfT2 = []
        listOfT3 = []
        listOfT4 = []
        
        for s in batch.state:
            listOfT2.append(torch.tensor(s, dtype = torch.float32).view(1, -1))
            
        state_batch = torch.cat([s for s in listOfT2 if s is not None])
        
        for s in batch.action:
            listOfT3.append(torch.tensor(s, dtype = torch.int64).view(1, -1))
        
        action_batch = torch.cat([s for s in listOfT3 if s is not None])
        
        for s in batch.reward:
            listOfT4.append(torch.tensor(s, dtype = torch.float32).view(1, -1))
            
        reward_batch = torch.cat([s for s in listOfT4 if s is not None])


        action_values = self.model(state_batch).to(device).gather(1,action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA).view(-1,1)


        loss = F.smooth_l1_loss(action_values.unsqueeze(0), expected_state_action_values.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()



    def load(self):
        state = torch.load(self.save_path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def save(self):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, self.save_path)

    
def play_game(player, render=True):
    # call this function to run your model on the environment
    # and see how it does
    num_episodes = 10
    for i in range(num_episodes):
        print('playing a new game')
        env = PongEnv()
        state = env.reset()
        action = player.get_action(state)
        done = False
        total_reward = 0
        while not done:
            next_state, reward, done, _ = env.step(action)
            if render:
                env.render()
            action = player.get_action(next_state)
            total_reward += reward
            memory.push(state, action, next_state, reward)
            p1.train()
            print(total_reward)
        player.save()

        


BATCH_SIZE = 50
GAMMA = 0.999
EPS_START = 0.9
EPS_DECAY = 200
EPS_END = 0.05
TARGET_UPDATE = 10


memory = ReplayMemory(10000)
optimizer = optim.RMSprop(MyModelClass().to(device).parameters())




    
    
p1 = PongPlayer('/Users/Joshua/Desktop/MLhackathon/pong_competition/out.txt')
play_game(p1)