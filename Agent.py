#BUILTIN PYTHON PACKAGES
import random
import copy
import math
from collections import namedtuple, deque

#OPEN SOURCE PYTHON PACKAGES
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

#CUSTOM PYTHON PACKAGES
import Model

###################################################################################################
###################################################################################################

#THE AGENT WHICH PERFORMS ACTIONS AND LEARNS THE ENVIRONMENT
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, load=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """

        #AGENT CONSTANTS
        BUFFER_SIZE = int(100000)  # replay buffer size
        BATCH_SIZE = 64         # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.LEARN_START = 100
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 10000
        self.TARGET_UPDATE = 20

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.0
        self.steps = 0

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE,self.device)

        #WEIGHT INITIALIZATION FUNCTION USED HERE JUST ONCE
        def init_weights(m):
            if isinstance(m, torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        #INITIAILIZE MODEL
        self.model = Model.DDQN(state_size=state_size,action_size=action_size)
        self.model_target = Model.DDQN(state_size=state_size,action_size=action_size)
        self.model.to(self.device)
        self.model_target.to(self.device)
        if load:
            self.model.load_state_dict(torch.load('model/ddqn.pth'))
            self.model_target.load_state_dict(torch.load('model/ddqn.pth'))
        else:
            self.model.apply(init_weights)
            self.model_target.apply(init_weights)
        self.opt = torch.optim.Adam(self.model.parameters(),lr=0.00001,betas=(0.0,0.9))

    #SAVE THE CURRENT MODEL
    def save_model():
        print("Model Saved...")
        torch.save(self.model.state_dict(),os.path.join('model','ddqn.pth'))

    #get an action for the current state using the current policy
    def act(self, state):
        self.model.eval()
        with torch.no_grad():
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps / self.EPS_DECAY)
            if sample > eps_threshold or True:
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                return self.model(state).max(1)[1].item()
            else:
                return random.randrange(3)

    #get the greedy action according to the current policy
    def act_greedy(self,state):
        self.model.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return self.model(state).max(1)[1].item()

    #get the worst action according to the current policy
    def act_dumb(self,state):
        self.model.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return self.model(state).min(1)[1].item()

    #optimize the current policy
    def optimize(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + ? * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        ======
        """
        self.model.train()
        experiences = self.memory.sample()
        s1, a, r, s2, d = experiences

        #get predicted q vals and target q vals
        qvals = self.model(s1)
        qvals = qvals.gather(1,a)
        with torch.no_grad():
            qvals_t = self.model_target(s2)
            qvals_t = qvals_t.max(1)[0].unsqueeze(1)

        #BELMONT EQUATION
        target_q = (qvals_t * self.GAMMA) * (1-d) + r

        #loss function to optimize
        loss = F.mse_loss(qvals,target_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        #UPDATE THE TARGET NETWORK AFTER A WHILE
        if self.steps % self.TARGET_UPDATE == 0:
            self.model_target.load_state_dict(self.model.state_dict())

        return loss.item()

    #METHOD FOR TAKING A STEP IN THE ENVIRONMENT DURING TRAINING
    def learn(self, experience):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.push(experience)
        loss = 0.0

        # Learn, if enough samples are available in memory
        if len(self.memory) >= self.LEARN_START:
            self.steps += 1
            loss = self.optimize()
        return loss


#REPLAY BUFFER USED BY THE AGENT TO STORE MEMORY
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size,device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = device
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = []
        self.pos = -1

    def push(self, experience):
        """Add a new experience to memory."""
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.pos = (self.pos + 1) % self.buffer_size

        state,action,reward,next_state,done = experience

        state = torch.Tensor(state)
        action = torch.Tensor([action])
        reward = torch.Tensor([reward])
        next_state = torch.Tensor(next_state)
        done = torch.Tensor([done])
        e = self.experience(state, action, reward, next_state, done)
        self.memory[self.pos] = e

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        batch = self.experience(*zip(*experiences))
        states = torch.stack(batch.state).float().to(self.device)
        actions = torch.stack(batch.action).long().to(self.device)
        rewards = torch.stack(batch.reward).float().to(self.device)
        next_states = torch.stack(batch.next_state).float().to(self.device)
        dones = torch.stack(batch.done).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



