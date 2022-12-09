import gym
import os 
import sys

# Add models
importpath = os.path.join(os.getcwd(), 'model')
sys.path.append(importpath)

from model import FrameEncoder

env = gym.make("ALE/Phoenix-v5")

# Simple Pytorch DQN algorithm, adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        # Use our general feature encoder
        self.encoder = FrameEncoder(env.action_space.n)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 16
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
