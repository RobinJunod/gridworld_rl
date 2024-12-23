#%% manual_play.py
import sys
import os

# Determine the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, project_root)

from environment.grid_env_1 import GridWorldEnv
#from environment.grid_env_0 import GridEnv_mask

print("Manual Play Mode")

def manual_play():
    # Initialize the environment
    env = GridWorldEnv(rows=9, cols=21, max_steps=40)
    state, _ = env.reset()
    
    print("Initial Environment:")
    env.render()
    
    done = False
    
    while not done:
        # Clear the output in terminal
        os.system('cls' if os.name == 'nt' else 'clear')
        
        
        try:
            action = input("Enter your action (0: Up, 1: Down, 2: Left, 3: Right, 4: Change to Left Mask, 5: Change to Middle Mask, 6: Change to Right Mask): ")
            action = int(action)
            if action not in range(7):
                raise ValueError
        except ValueError:
            print("Invalid action. Please enter a number between 0 and 6.")
            continue
        
        # Apply the action to the environment
        state, reward, done, truncated, info = env.step(action)
        
        # Display the current step information
        print(f"\nStep: {env.current_step}")
        print(f"Action Taken: {action}")
        print(f"Reward Received: {reward}")
        target_visible = state[env.target_pos[0], env.target_pos[1], 1] == 1
        print(f"Is Target Visible: {'Yes' if target_visible else 'No'}")
        
        # Visualize the updated grid
        env.render()
        
        if done:
            if reward > 0:
                print("Congratulations! You've reached the target.")
            else:
                print("Maximum steps reached without reaching the target.")
            break

if __name__ == "__main__":
    manual_play()
# %%
