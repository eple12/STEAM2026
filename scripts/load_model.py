from stable_baselines3 import DQN
import gymnasium as gym
import highway_env
from utils import *
from tqdm.notebook import trange

gym.register_envs(highway_env)

model = DQN.load(r"C:\Users\user\Desktop\WorkSpace\STEAM2026\models\highway\initial-better.zip")

env = gym.make('highway-fast-v0', render_mode='rgb_array')
env = record_videos(env)
for episode in trange(3, desc='Test episodes'):
    (obs, info), done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
env.close()
show_videos()