#%% src/agent/train_agent.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, project_src)

import numpy as np
import torch.nn as nn
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback

from environment.grid_env_1 import GridWorldEnv
from config import (
    ROWS, COLS, MAX_STEPS, INIT_DIST, INIT_DISTR, TIMESTEPS,
    TENSORBOARD_LOG, MODEL_SAVE_PATH
)
from datetime import datetime

# Update TENSORBOARD_LOG with timestamp to create a unique directory for each run
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
TENSORBOARD_LOG_DIR = os.path.join(TENSORBOARD_LOG, timestamp)
CHECKPOINT_DIR = os.path.join(TENSORBOARD_LOG, f"checkpoints_{timestamp}")

# Ensure the directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# Callback to save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=200_000,  # Save every n steps
    save_path=CHECKPOINT_DIR,  # Directory to save the checkpoints
    name_prefix='ppo_grid_env',  # Prefix for checkpoint files
    save_replay_buffer=False,  # Whether to save the replay buffer
    save_vecnormalize=False  # Whether to save VecNormalize statistics
)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.successes = deque(maxlen=20)
        self.episodes = 0

    def _on_step(self) -> bool:
        # Access the 'dones' and 'infos' from the current step
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if dones is True and infos is not None:
            done = dones[0]
            info = infos[0]
            if done:
                is_success = info['is_success']
                # Success if terminated=True and truncated=False
                success = 1 if is_success else 0
                self.successes.append(success)

                # Calculate success rate over the last 'window' episodes
                if len(self.successes) > 0:
                    success_rate = sum(self.successes) / len(self.successes)
                    self.logger.record("rollout/success_rate_last_20", success_rate)
                    if self.verbose > 0:
                        print(f"Success Rate (last {self.window}): {success_rate:.2f}")
        return True




def train():
    print("Init the env ...")
    # Instantiate the environment
    env = GridWorldEnv(rows=ROWS, cols=COLS, max_steps=MAX_STEPS, num_distractors=INIT_DIST)
    print("Init the model ...")
    # Define the PPO model
    model = PPO(
        "MlpPolicy",  # Convolutional neural network policy
        env,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.ReLU
        ),
        gamma=0.95,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        ent_coef=0.01,
    )

    print("Start training ...")
    # Train the model
    callback = CallbackList([TensorboardCallback(), checkpoint_callback])
    model.learn(total_timesteps=TIMESTEPS, 
                callback=callback)
    print("Training completed.")
    # Save the trained model
    current_time = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    model_save_path = os.path.join(MODEL_SAVE_PATH, f"ppogrid_{current_time}_initial_dist_{INIT_DIST}_initial_distr_{INIT_DISTR}")
    model.save(model_save_path)

    print("Model saved.")

if __name__ == "__main__":
    print("Training the agent...")
    train()

# Tensorboard command : 
# tensorboard --logdir=data/logs/tensorboard --port=6006





#%% 

