#%%
import sys
import os

# Determine the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, project_root)



import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import logging

# Configure a module-specific logger without setting the root logger's level
logger = logging.getLogger(__name__)

# Constants for channels
AGENT = 0
TARGET = 1
DISTRACTORS = 2
CURRENT_MASK_POS = 3
PREVIOUS_MASK_POS = 4

# Mask Positions
MASK_POSITIONS = ['left', 'middle', 'right']

# Action Definitions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_SHIFT_MASK_LEFT = 4
ACTION_SHIFT_MASK_MIDDLE = 5
ACTION_SHIFT_MASK_RIGHT = 6
NUM_ACTIONS = 7

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class GridWorldEnv(gym.Env):
    """
    Custom Grid World Environment with Mask Visibility.

    Observation:
        Type: Box(9, 21, 5)
        Num     Observation                Min     Max
        0       Agent Position             0       1
        1       Target Position            0       1
        2       Distractors                0       1
        3       Current Mask Position      0       1
        4       Previous Mask Position     0       1

    Actions:
        Type: Discrete(7)
        Num   Action
        0     Move Up
        1     Move Down
        2     Move Left
        3     Move Right
        4     Shift Mask to Left
        5     Shift Mask to Middle
        6     Shift Mask to Right

    Reward:
        +1.0 for reaching the target.
        -0.01 per step taken.

    Starting State:
        Agent and target placed randomly within grid constraints.

    Episode Termination:
        Agent reaches the target.
        Agent exceeds max_steps.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, rows=9, cols=21, max_steps=80, num_distractors=20):
        super(GridWorldEnv, self).__init__()

        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.num_distractors = num_distractors
        
        self.current_step = 0

        # Define action and observation space
        # Actions: 0-3 movement, 4-6 mask shifts
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation space: 9x21 grid with 5 channels
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(5, self.rows, self.cols),
            dtype=np.uint8
        )

        # Initialize the grid
        self.grid = np.zeros((self.rows, self.cols), dtype=object)

        # Initialize positions
        self.agent_pos = None
        self.target_pos = None
        self.distractors = []

        # Initialize mask positions
        self.current_mask_pos = 1  # Start with 'middle' takes [0,1,2]
        self.previous_mask_pos = 1  # No movement at reset

        # Seed for reproducibility
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, max_distance=3):
        super().reset(seed=seed)
        self.current_step = 0

        # Clear the grid
        self.grid = np.zeros((self.rows, self.cols), dtype=object)

        # Place the agent randomly
        self.agent_pos = [
            self.np_random.integers(self.rows),
            self.np_random.integers(self.cols)
        ]
        self.grid[self.agent_pos[0], self.agent_pos[1]] = "Agent"
        
        # Place the target randomly (option:inside min dist), ensuring it doesn't overlap with the agent
        while True:
            self.target_pos = [
                self.np_random.integers(self.rows),
                self.np_random.integers(self.cols)
            ]
            distance = manhattan_distance(self.agent_pos, self.target_pos)
            if self.target_pos != self.agent_pos and distance <= max_distance:
                break

        self.grid[self.target_pos[0], self.target_pos[1]] = "Target"

        # Place distractors
        self.distractors = []
        for _ in range(self.num_distractors):
            attempts = 0
            while attempts < 100:
                pos = [
                    self.np_random.integers(self.rows),
                    self.np_random.integers(self.cols)
                ]
                if (self.grid[pos[0], pos[1]] ==0):
                    self.grid[pos[0], pos[1]] = "Distractor"
                    self.distractors.append(pos)
                    break
                attempts += 1
            if attempts >= 100:
                logger.warning("Max attempts reached while placing distractors.")
                continue  # Skip placement if max attempts reached

        # Initialize mask positions
        self.previous_mask_pos = 1
        self.current_mask_pos = 1  # Start with 'middle'

        return self.get_observation(), {}

    def get_observation(self):
        """
        Generate the current observation based on the mask position.
        """
        obs = np.zeros((5, self.rows, self.cols), dtype=np.uint8)

        # Define mask range based on current_mask_pos
        mask_size = self.cols//3  # Width of the mask
        if self.current_mask_pos == 0:  # Left
            mask_start = 0
        elif self.current_mask_pos == 1:  # Middle
            mask_start = (self.cols - mask_size) // 2
        else:  # Right
            mask_start = self.cols - mask_size
        mask_end = mask_start + mask_size

        # Adjust mask to fit within grid boundaries
        mask_start = max(0, mask_start)
        mask_end = min(self.cols, mask_end)

        # Channel 0: Agent Position (always visible)
        obs[AGENT, self.agent_pos[0], self.agent_pos[1]] = 1.0

        # Channel 1: Target Position (visible only if within mask)
        if (self.target_pos[0] < self.rows and
            mask_start <= self.target_pos[1] < mask_end):
            obs[TARGET, self.target_pos[0], self.target_pos[1]] = 1.0

        # Channel 2: Distractors (visible only if within mask)
        for distractor in self.distractors:
            if (distractor[0] < self.rows and
                mask_start <= distractor[1] < mask_end):
                obs[DISTRACTORS, distractor[0], distractor[1]] = 1.0

        # Channel 3: Current Mask Position (1s where visible, 0 otherwise)
        obs[CURRENT_MASK_POS, :, mask_start:mask_end] = 1.0

        # Channel 4: Previous Mask Position (1s where previously visible, 0 otherwise)
        if self.previous_mask_pos is not None:
            if self.previous_mask_pos == 0:  # Left
                prev_start = 0
            elif self.previous_mask_pos == 1:  # Middle
                prev_start = (self.cols - mask_size) // 2
            else:  # Right
                prev_start = self.cols - mask_size
            prev_end = prev_start + mask_size
            prev_start = max(0, prev_start)
            prev_end = min(self.cols, prev_end)
            obs[PREVIOUS_MASK_POS, :, prev_start:prev_end] = 1.0

        return obs

    def step(self, action):
        reward = -0.02 # Time penalty
        terminated = False
        is_success = False
        # Compute old distance
        old_distance = manhattan_distance(self.agent_pos, self.target_pos)

        # Agent performs action
        self.current_step += 1
        if action == ACTION_UP:
            self.move_agent(-1, 0)
        elif action == ACTION_DOWN:
            self.move_agent(1, 0)
        elif action == ACTION_LEFT:
            self.move_agent(0, -1)
        elif action == ACTION_RIGHT:
            self.move_agent(0, 1)
        elif action == ACTION_SHIFT_MASK_LEFT:
            self.shift_mask(0)
        elif action == ACTION_SHIFT_MASK_MIDDLE:
            self.shift_mask(1)
        elif action == ACTION_SHIFT_MASK_RIGHT:
            self.shift_mask(2)
        else: 
            print("Invalid action. Please enter a number between 0 and 6.")

        # Generate new observation
        obs = self.get_observation()

        if not self._is_target_visible():
            reward -= 0.5
            
        # Reward function: Visibility + distance improvement
        if self._is_target_visible() and manhattan_distance(self.agent_pos, self.target_pos) < old_distance:
            reward += 0.5 * 1/(1+(manhattan_distance(self.agent_pos, self.target_pos)))

        # End-of-episode reward
        if  self._is_target_visible()and self.agent_pos == self.target_pos:
            reward += 5.0
            terminated  = True
            is_success = True
            logger.info(f"Agent reached the target at {self.target_pos} in {self.current_step} steps.")

        # Check for step limit
        if self.current_step >= self.max_steps:
            terminated  = True
            is_success = False
            logger.info(f"Agent did not reach the target within {self.max_steps} steps.")

        # Include terminated and truncated flags in the info dictionary
        info = {
            "is_success": is_success,
        }

        return obs, reward, terminated, False, info

    def move_agent(self, delta_row, delta_col):
        """
        Move the agent in the specified direction.
        """
        new_row = self.agent_pos[0] + delta_row
        new_col = self.agent_pos[1] + delta_col

        # Ensure new position is within bounds
        new_row = np.clip(new_row, 0, self.rows - 1)
        new_col = np.clip(new_col, 0, self.cols - 1)

        # Update agent position
        self.agent_pos = [new_row, new_col]
        logger.info(f"Agent moved to {self.agent_pos}")

    def shift_mask(self, position):
        """
        Shift the mask window to a specified position.
        :param position: 0 for 'left', 1 for 'middle', 2 for 'right'
        """
        if position in [0, 1, 2]:
            self.previous_mask_pos = self.current_mask_pos
            self.current_mask_pos = position
            logger.info(f"Mask shifted to {MASK_POSITIONS[self.current_mask_pos]} position.")
        else:
            logger.warning("Invalid mask shift position. Choose from 0 (left), 1 (middle), 2 (right).")

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only 'human' render mode is supported.")

        # Define mask range based on current_mask_pos
        mask_size = 7  # Width of the mask
        if self.current_mask_pos == 0:  # Left
            mask_start = 0
        elif self.current_mask_pos == 1:  # Middle
            mask_start = (self.cols - mask_size) // 2
        else:  # Right
            mask_start = self.cols - mask_size
        mask_end = mask_start + mask_size

        # Build the entire grid visualization
        grid_visual = ""
        for r in range(self.rows):
            for c in range(self.cols):
                cell = ". "  # Default empty
                if [r, c] == self.agent_pos:
                    cell = "a "  # Agent
                elif [r, c] == self.target_pos:
                    cell = "t "  # Target
                elif [r, c] in self.distractors:
                    cell = "d "  # Distractor

                # Highlight mask visibility
                if mask_start <= c < mask_end:
                    cell = cell.upper()  # Indicate visibility with bigger letter

                grid_visual += cell
            grid_visual += "\n"
        print(grid_visual)

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass

    def _is_target_visible(self):
        # Define logic based on mask position
        mask_size = 7
        if self.current_mask_pos == 0:
            mask_start = 0
        elif self.current_mask_pos == 1:
            mask_start = (self.cols - mask_size) // 2
        else:
            mask_start = self.cols - mask_size
        mask_end = mask_start + mask_size
        return (mask_start <= self.target_pos[1] < mask_end)


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    env = GridWorldEnv(rows=9, cols=21, max_steps=40)
    check_env(env)
    
    state, _ = env.reset()
    print(state)
    print(state.shape)
# %%
