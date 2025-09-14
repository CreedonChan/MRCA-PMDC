"""
This is for the single_int robot map config
time: 20250704

Introduction: This is for the 11 obs with the fixed scenario
"""


import numpy as np

from config.scenario.obs_init_2D import rectangle_obs_init_2d
from config.scenario.robot_start_pos_init_2D_2 import robot_init_2d
from multirobot.modules.system import System
from multirobot.visualizer.multi_robot_visualizer import MultiRobotVisualizer

# basic config
dim = 2
dt = 0.1
ws = np.array([[-5, 5], [-5, 5]])       # [[x1, x2], [y1, y2]]

# collision_threshold
collision_threshold = 0.1

# weight
# had put it into the single_int model

# the num of the robot and obs
# here the num is only for the circle shape of the init pos of robots
n_robot = 8     # mode
n_obs = 11       # here we only have the 0, 1, 3 num of the circle obs

ROBOT_DATABASE = {
    'basic_config': {
        'dim': dim,
        'dt': dt,
        'N': 15,
        'speed_max': 0.4,
        'bound_radius': 1.5,
        'bound_global': ws,
        'collision_threshold': collision_threshold,
    },
    'robots_config': robot_init_2d()
}

OBSTACLE_DATABASE = {
    'basic_config': {
        'dim': dim,
    },
    'obs_config': rectangle_obs_init_2d(n_obs)
}

SYSTEM_DATABASE = {
    'dim': dim,
    'ws': ws,
    'dt': dt,
}

multi_robot_keys: list = list(ROBOT_DATABASE['robots_config'].keys())
multi_obs_keys: list = list(OBSTACLE_DATABASE['obs_config'].keys())

def initialize_system_and_visualizer(visualizer_init: bool=True):
    system = System(robot_database=ROBOT_DATABASE,
                    obs_database=OBSTACLE_DATABASE,
                    **SYSTEM_DATABASE)

    # visualizer init
    visualizer = None
    if visualizer_init:
        visualizer = MultiRobotVisualizer(ws.tolist())
        for i in multi_robot_keys:
            robot_init_data = ROBOT_DATABASE['robots_config'][i]
            visualizer.add_robot(i, robot_init_data['pos_real'], robot_init_data['radius'], show_orientation=False)

        for i in multi_obs_keys:
            obs_init_data = OBSTACLE_DATABASE['obs_config'][i]
            visualizer.add_rectangle_obstacle(i, obs_init_data['pos_real'], obs_init_data['size'], obs_init_data['yaw'])

    return system, visualizer
