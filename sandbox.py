''' Test file for playing around with everything '''

import gym
from gym import envs
import os 
import sys

# Add models
importpath = os.path.join(os.getcwd(), 'model')
sys.path.append(importpath)

from model import FrameEncoder

# print(envs.registry.all())
env = gym.make("Phoenix-v4")

print(env.action_space.n)
