from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

import matplotlib.pyplot as plt
import mlagents
from PIL import Image
from mlagents.envs import UnityEnvironment
import sys
print("Python version:")
print(sys.version)
from mlagents.envs import UnityEnvironment

class xSpace(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec =( array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='fire') ,
        array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='move') )
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(39,), dtype=np.float64, minimum=-1.1,maximum=1.1, name='observation')
    self._state = 0
    self._episode_ended = False
    env_name = "C:\\Users\\matt\\Haskell\\xUnity\\machine10\\ml-agents-master\\xLaser\\laserDefender.exe"  # Name of the Unity environment binary to launch
    self._train_mode = True  # Whether to run the environment in training or inference mode
    env = UnityEnvironment(file_name=env_name)
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]
    aSize= brain.vector_observation_space_size
    self._env=env
    print(env.reset_parameters)
    self._default_brain=default_brain
    self._brain=brain
    self._aSize=aSize
    self.MaxSteps=50*200
    self._reward=0
    self._counter=0
    self._CumReward=0
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec
  def getEpisodeEnded(self):
      return self._episode_ended
  def _reset(self):
    self._state = 0
    self._episode_ended = False
    self._env.reset(train_mode=self._train_mode)[self._default_brain]
    self._reward=0
    self._state= np.zeros((39),dtype=np.float64)
    self._counter=0
    #print('reset')
    return ts.restart(self._state )

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      print("reward:"+str(self._CumReward) )
      self._CumReward=0
      return self.reset()
    #print("action:"+str(action)+"   type:"+str(type(action)) )
    k1=action[0]
    #print("  val1:"+str(k1)+"   tpye:"+str(type(k1)) + "   shape:"+str(k1.shape) )
    s1=k1[()]
    k2=action[1]
    s2=k2[()]
    #print("  val1:"+str(s1)+"   tpye:"+str(type(s1)) )
    env_info = self._env.step([s1,s2])[self._default_brain]
    self._episode_ended= env_info.local_done[0]
    # Make sure episodes don't go on forever.
    #self._state= np.zeros((39),dtype=np.float32)
    self._state=env_info.vector_observations[0]
    #print('dtype:'+str(type(self._state)) +'   shape:'+str(self._state.shape )+'  element:'+str(type(self._state[0])) )
    self._reward=env_info.rewards[0]
    self._CumReward+=self._reward
    self._counter+=1
    #print("counter"+str(self._counter) )
    if self._episode_ended or self._counter >= self.MaxSteps:
      #print("reward:"+str(self._reward))
      return ts.termination(self._state, self._reward)
    else:
      return ts.transition(self._state, reward=self._reward, discount=1.0)
#environment = xSpace()
#utils.validate_py_environment(environment, episodes=5)
#print("success!")
