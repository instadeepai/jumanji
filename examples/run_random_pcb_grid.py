import argparse

import numpy as np

from jumanji.pcb_grid import PcbGridEnv

NUM_AGENTS = 4
SIZE = 12
DIFFICULTY = "easy"
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=SIZE, type=int)
    parser.add_argument("--nets", default=NUM_AGENTS, type=int)
    parser.add_argument("--diff", default=DIFFICULTY)
    args = parser.parse_args()

    env = PcbGridEnv(args.size, args.size, args.nets, args.diff)
    env.reset()

    for _ in range(50):
        actions = {agent_id: np.random.randint(1, 5) for agent_id in range(args.nets)}
        obs, reward, done, info = env.step(actions)
        env.render("human")
        if done["__all__"]:
            env.reset()
            print("Done")
    env.close()

    input("Finished. Press Enter to quit.")
