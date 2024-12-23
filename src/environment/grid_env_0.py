import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Determine the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, project_root)

from config import (
    ROWS, COLS, MAX_STEPS, INIT_DIST, INIT_DISTR, TIMESTEPS,
    TENSORBOARD_LOG, MODEL_SAVE_PATH
)

def manhattan_distance(pos1, pos2):
   return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class GridEnv_mask(gym.Env):
    def __init__(self, rows=ROWS, cols=COLS, max_steps=MAX_STEPS):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.dim_env_action = 7
        self.max_steps = max_steps
        self.current_step = 0

        # Updated state dimensions: 4 channels
        # 0: agent, 1: mask, 2: shape: square (1)/circle (-1), 3: color: blue (1)/red (-1)
        self.state = np.zeros((rows, cols, 4), dtype=np.float32)

        self._abs_step = 0
        self._success_count = []
        self._max_dist = INIT_DIST
        self._max_distr = INIT_DISTR
        self.switch = 1

        # Updated observation space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(rows, cols, 4), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.dim_env_action)

        # Store the object layout
        # We'll keep track of object types in a separate array for quick re-application of the mask
        # object_grid will store tuples (square_or_circle, color) or None if empty:
        # square_or_circle in {"square", "circle"}
        # color in {"blue", "red"}
        # For the target (blue square), we know it's "square", "blue".
        self.object_grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]

        self.reset()

    def reset(self, seed=None, options=None):

        # Curriculum learning
        if (self._abs_step > 201) and (np.mean(self._success_count[-200:]) > 0.8):
            if (self._max_dist < (ROWS + COLS)) and (self.switch > 0):
                self._max_dist += 1
                self._abs_step = 0
                if (self._max_distr < (self.rows/3 * self.cols/3 - 1)):
                    self.switch =  -1 * self.switch

        if (self._abs_step > 201) and (np.mean(self._success_count[-200:]) > 0.8):
            if (self._max_distr < (self.rows/3 * self.cols/3 - 1)) and (self.switch < 0):
                self._max_distr += 1
                self._abs_step = 0
                if (self._max_dist < (ROWS + COLS)):
                    self.switch =  -1 * self.switch

        self._abs_step += 1

        # Handle the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.state.fill(0)
        self.object_grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]

        # Place the agent randomly
        self.agent_pos = [np.random.randint(self.rows), np.random.randint(self.cols)]

        # We have 21 blocks of size 3x3. Blocks arrangement:
        # Rows of blocks: 3 (each 3 rows)
        # Cols of blocks: 7 (each 3 columns)
        # Total blocks: 3*7 = 21
        # Choose one block for the target (blue square):
        block_indices = list(range(21))
        np.random.shuffle(block_indices)
        target_block = block_indices[0]
        distractor_blocks = block_indices[(21-self._max_distr):]

        # Compute the block row and col for the target
        target_block_row = target_block // 7  # integer division
        target_block_col = target_block % 7

        # The top-left corner of this block:
        block_row_start = target_block_row * 3
        block_col_start = target_block_col * 3

        # Choose a random cell within this 3x3 block for the target
        tr = block_row_start + np.random.randint(3)
        tc = block_col_start + np.random.randint(3)
        
        # Define the target position
        self.target_pos = [tr, tc]

        # If we want to ensure distance constraints, we can re-select until conditions meet.
        # We want that the target is within the _max_dist
        while (self.agent_pos == self.target_pos) or (manhattan_distance(self.agent_pos, self.target_pos) > self._max_dist):
            # Re-roll target position within the same block if constraints not met
            np.random.shuffle(block_indices)
            target_block = block_indices[0]
            distractor_blocks = block_indices[(21-self._max_distr):]

            # Compute the block row and col for the target
            target_block_row = target_block // 7  # integer division
            target_block_col = target_block % 7

            # The top-left corner of this block:
            block_row_start = target_block_row * 3
            block_col_start = target_block_col * 3

            # Choose a random cell within this 3x3 block for the target
            tr = block_row_start + np.random.randint(3)
            tc = block_col_start + np.random.randint(3)
            self.target_pos = [tr, tc]

        # Used for the reward system
        self.initial_distance = manhattan_distance(self.agent_pos, self.target_pos)
        self.min_distance = self.initial_distance


        # Place the target: blue square
        self.object_grid[self.target_pos[0]][self.target_pos[1]] = ("square", "blue")

        # Now place distractors in the remaining blocks
        # Each distractor block: choose random cell and random type from {red square, blue circle, red circle}
        distractor_types = [
            ("square", "red"),
            ("circle", "blue"),
            ("circle", "red")
        ]

        for db in distractor_blocks:
            db_row = db // 7
            db_col = db % 7
            r_start = db_row * 3
            c_start = db_col * 3

            # Choose a random cell in this 3x3 block
            rr = r_start + np.random.randint(3)
            cc = c_start + np.random.randint(3)

            # Ensure we don't place on target or agent (should not happen but just in case)
            # If conflicts occur, just reroll a few times
            for _ in range(10):
                if [rr, cc] != self.target_pos and [rr, cc] != self.agent_pos:
                    break
                rr = r_start + np.random.randint(3)
                cc = c_start + np.random.randint(3)

            # Choose a random distractor type
            d_type = distractor_types[np.random.randint(len(distractor_types))]
            self.object_grid[rr][cc] = d_type

        # Initialize mask value
        self.mask = np.random.randint(3)

        if self._max_distr >= 1:
            # Check if any distractor is within self._max_dist from the agent
            agent_r, agent_c = self.agent_pos
            any_distr_close = False
            # Convert object_grid to a NumPy array for vectorized operations
            obj_array = np.array(self.object_grid, dtype=object)

            # Extract all positions where there is an object (not None)
            distractor_positions = np.argwhere(obj_array != None)

            # Filter out the target position
            target_r, target_c = self.target_pos
            mask_not_target = (distractor_positions[:,0] != target_r) | (distractor_positions[:,1] != target_c)
            distractor_positions = distractor_positions[mask_not_target]

            # Compute Manhattan distances to the agent
            agent_r, agent_c = self.agent_pos
            distances = np.abs(distractor_positions[:,0] - agent_r) + np.abs(distractor_positions[:,1] - agent_c)

            # Check if any distractor is within INIT_DIST
            any_distr_close = np.any(distances <= INIT_DIST)

            # If no distractor is within INIT_DIST, add one now
            if not any_distr_close:
                distractor_types = [
                    ("square", "red"),
                    ("circle", "blue"),
                    ("circle", "red")
                ]
                # Attempt to place a distractor within INIT_DIST range of the agent
                for _ in range(100):  # limit tries to avoid infinite loop
                    rr = agent_r + np.random.randint(-INIT_DIST, INIT_DIST + 1)
                    cc = agent_c + np.random.randint(-INIT_DIST, INIT_DIST + 1)
                    if (0 <= rr < self.rows and 0 <= cc < self.cols and
                        [rr, cc] != self.agent_pos and [rr, cc] != self.target_pos and
                        self.object_grid[rr][cc] is None):
                        d_type = distractor_types[np.random.randint(len(distractor_types))]
                        self.object_grid[rr][cc] = d_type
                        break

        # Apply the accessibility of the information through the field of view:
        self.apply_mask()

        return self.state, {}

    @property
    def max_dist(self):
        return self._max_dist

    @max_dist.setter
    def max_dist(self, value):
        self._max_dist = value

    @property
    def max_distr(self):
        return self._max_distr

    @max_distr.setter
    def max_distr(self, value):
        self._max_distr = value

    @property
    def success_count(self):
        return self._success_count

    @success_count.setter
    def success_count(self, value):
        self._success_count = value

    def apply_mask(self):
        # Clear state
        self.state.fill(0)

        # Define visibility ranges based on mask value
        # mask 0: columns 0-6 visible
        # mask 1: columns 7-13 visible
        # mask 2: columns 14-20 visible
        ranges = [(0, 7), (7, 14), (14, 21)]
        start, end = ranges[self.mask]

        # Set mask visibility in [start:end] columns
        self.state[:, start:end, 1] = 1  # mask channel

        # Place the agent (always visible, regardless of mask)
        self.state[self.agent_pos[0], self.agent_pos[1], 0] = 1

        # Place objects only if their column is within the visible mask range
        for r in range(self.rows):
            for c in range(self.cols):
                if self.object_grid[r][c] is not None:
                    # object_grid[r][c]: (type, color)
                    # type in {"square", "circle"}
                    # color in {"blue", "red"}
                    # Check if visible
                    if start <= c < end:
                        o_type, o_color = self.object_grid[r][c]
                        if o_type == "square":
                            self.state[r, c, 2] = 1
                            if o_color == "blue":
                                self.state[r, c, 3] = 1
                            else:
                                self.state[r, c, 3] = -1
                        else:
                            self.state[r, c, 2] = -1
                            if o_color == "blue":
                                self.state[r, c, 3] = 1
                            else:
                                self.state[r, c, 3] = -1

    def step(self, action):

        self.current_step += 1
        old_agent_pos = self.agent_pos.copy()

        # Movement actions (0=up,1=down,2=left,3=right)
        if action in [0, 1, 2, 3]:
            if action == 0:
                self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            elif action == 1:
                self.agent_pos[0] = min(self.rows - 1, self.agent_pos[0] + 1)
            elif action == 2:
                self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
            elif action == 3:
                self.agent_pos[1] = min(self.cols - 1, self.agent_pos[1] + 1)

        # Mask-changing actions (4 left,5 middle,6 right)
        elif action in [4, 5, 6]:
            self.mask = action - 4

        # After moving or changing mask, re-apply the mask
        self.apply_mask()

        # Check if target is visible
        # The target is a blue square at self.target_pos.
        # Target visible if mask covers its column and channels indicate square=1, blue=1
        tr, tc = self.target_pos
   
        target_is_visible = (self.state[tr, tc, 2] == 1 and   # square
                             self.state[tr, tc, 3] == 1)       # blue
    
        # Compute reward and done conditions
        old_distance = manhattan_distance(old_agent_pos, self.target_pos)
        new_distance = manhattan_distance(self.agent_pos, self.target_pos)


        done = self.current_step >= self.max_steps
        reward = 0

        if self.current_step >= self.max_steps:
            self._success_count.append(0)

        # Required by gym standard package protocol
        truncated = done

        # If the agent get closer to the target, reward proportional to proximity and limited to new best performance
        if target_is_visible and old_distance > new_distance and self.min_distance > new_distance:
            reward = 1 / (new_distance+1)
            self.min_distance = new_distance


        # If the agent reaches the target position while it is visible
        if target_is_visible and self.agent_pos == self.target_pos:# and action == 7:
            done = True
            reward = 1
            self._success_count.append(1)

        info = {}

        return self.state, reward, done, truncated, info

    def render(self, mode='human'):
        print(f"Agent position: {self.agent_pos}, Target position: {self.target_pos}")

    def close(self):
        pass