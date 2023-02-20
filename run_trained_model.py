import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name = "CarRacing-v0"
env = gym.make(environment_name)

ppo_path = os.path.join('PPO_200k_Driving_model')
model = PPO.load(ppo_path, env)

evaluate_policy(model, env, n_eval_episodes=1, render=True)
env.close()
