#%% tests/test_environment.py
import sys
import os
import unittest
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.environment.grid_env_0 import GridEnv_mask

class TestGridEnv(unittest.TestCase):
    def setUp(self):
        self.env = GridEnv_mask(rows=9, cols=21, max_steps=40, init_dist=1, init_distr=1)

    def test_reset(self):
        state, info = self.env.reset()
        self.assertEqual(state.shape, (9, 21, 4))
        self.assertIsInstance(info, dict)

    def test_step(self):
        self.env.reset()
        state, reward, done, truncated, info = self.env.step(0)  # Action: Up
        self.assertEqual(state.shape, (9, 21, 4))
        self.assertIn(reward, [0, 1, 1/(1+1)])  # Based on step logic
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

if __name__ == '__main__':
    unittest.main()


# %%
