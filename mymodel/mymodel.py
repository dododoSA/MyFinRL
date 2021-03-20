# common library
import pandas as pd
import numpy as np
import time
import gym

from .tfjdrl import TFJDRL

class My_Agent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
    train_PPO()
        the implementation for PPO algorithm
    train_A2C()
        the implementation for A2C algorithm
    train_DDPG()
        the implementation for DDPG algorithm
    train_TD3()
        the implementation for TD3 algorithm
    train_SAC()
        the implementation for SAC algorithm
    DRL_prediction()
        make a prediction in a test dataset and get results
    """

    @staticmethod
    def DRL_prediction(model, test_data, test_env, test_obs):
        """make a prediction"""
        start = time.time()
        account_memory = []
        actions_memory = []
        model.eval()
        with torch.no_grad():
            for i in range(len(test_data.index.unique())):
                action, _states = model(test_obs)
                test_obs, rewards, dones, info = test_env.step(action)
                if i == (len(test_data.index.unique()) - 2):
                    account_memory = test_env.env_method(method_name="save_asset_memory")
                    actions_memory = test_env.env_method(method_name="save_action_memory")
        end = time.time()
        return account_memory[0], actions_memory[0]


    def __init__(self, env):
        self.env = env
        self.model = TFJDRL()


    def train_model(self, model, tb_log_name, total_timesteps=5000):
        timesteps = 0

        while timesteps < total_timesteps:
            obs = self.env.reset()
            done = False
            while not done:


            timesteps += 1
        
        return model
