# src/main.py

import argparse
from agent.train_agent import train
from src.environment.grid_env_0 import GridEnv_mask
from visualization.plot_grid import plot_grid
from visualization.manual_play import manual_play

def main():
    parser = argparse.ArgumentParser(description="Grid World RL Project")
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--visualize', action='store_true', help='Visualize agent performance')
    parser.add_argument('--self_play', action='store_true', help='Manual self-play')
    args = parser.parse_args()

    if args.train:
        train()
    elif args.visualize:
        plot_grid()
    elif args.self_play:
        manual_play()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()