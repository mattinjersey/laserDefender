# laserDefender
## Introduction
I have been studying reinforcement learning. 
The idea here is to teach a machine learning to play a game.
The way that the machine learning will learn is to play the game over and over again, trying to maximize the score.

The underlying algorithm it will use is PPO learning, a type of reinforcement learning.

## What this repo contains
This repo contains a Unity game, called Space Shooter. The game is designed to be controlled by machine learning. So the machine learning can control the player to maximize the score.
So you are going to need Unity on your computer to get going with this. (its free!)

The repo uses ML-agents to control the player. You can install that from Github.
That is free. But you will have to read the instructions for ML-agents to learn how to use it.

The repo also tries using TF-agents to control the player. 
That's optional. TF-agents is a reinforcement library that substantially overlaps ML.
But if you want to try it, you will need to install tf-agents from GitHub.

## Credit and Links.
The Space Shooter game is derived from a game in a course that I took. Here is a link to the _Udemy_ course: [Unity Course](https://www.udemy.com/unitycourse/). Great course and I recommend it!
I've modified the game so that it can communicate with Python. This enables us to control the player and use machine learning to get a high score. Also I modified the game to make it more challenging so that the machine learning wouldn't find a trivial solution.

To train the game with machine learning, you will need to install ML agents: 
[ML agents](https://github.com/Unity-Technologies/ml-agents)

I also tried the reinforcement library TF Agents:
[TF agents](https://github.com/tensorflow/agents)

## Instructions
You will need Unity on your computer. Go to the Unity website to download and install it, for free.
Now you should be able to start and play the Unity game in the repo. If you have some problems, you may need to go to the GitHub/MLagents to install MLagents.
MLagents is a great system that enables Unity to communicate with python.


