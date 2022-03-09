import argparse
from typing import Tuple

import numpy as np

from jumanji.pcb_grid.pcb_grid import PcbGridEnv
from jumanji.pcb_grid.pcb_grid_viewer import PcbGridViewer

NUM_AGENTS = 4
SIZE = 32
DIFFICULTY = "easy"
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=SIZE, type=int)
    parser.add_argument("--nets", default=NUM_AGENTS, type=int)
    parser.add_argument("--diff", default=DIFFICULTY)
    args = parser.parse_args()

    env = PcbGridEnv(args.size, args.size, args.nets, args.diff)
    env_viewer = PcbGridViewer(env, 1000, 1000)  # Viewer to render environment
    env.reset()

    for _ in range(1000):
        actions = {agent_id: np.random.randint(1, 5) for agent_id in range(args.nets)}
        env_tuple: Tuple = env.step(actions)
        obs, reward, done, _ = env_tuple
        env_viewer.render_with_mode(mode="fast")  # Render environment
        if done["__all__"]:
            env.reset()
            print("Done")
    env_viewer.close()

    input("Finished. Press Enter to quit.")
