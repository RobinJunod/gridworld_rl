# config.py
import sys
import os

# Determine the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))

ROWS = 9
COLS = 21
MAX_STEPS = 40
INIT_DIST = 1  # Initial distance for curriculum learning
INIT_DISTR = 10  # Initial number of distractors for curriculum learning
TIMESTEPS = 20000000
TENSORBOARD_LOG = project_root + "/data/logs/tensorboard/"
MODEL_SAVE_PATH = project_root + "/models/saved_models/"