#!/usr/bin/env python3
import sys
import os

# print('\n\n\n')
# print("Current working directory:", os.getcwd())

# print("Directory contents:", os.listdir())
# print("sys.path before adjustment:", sys.path)
# print('\n\n\n')

# # Add the directory containing robobo_interface to sys.path
# sys.path.append(r'C:\Users\Mohammed\Desktop\learning_machines\learning_machines_robobo-master\examples\full_project_setup\catkin_ws\src')
# print("sys.path after adjustment:", sys.path)

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions


if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    run_all_actions(rob)
