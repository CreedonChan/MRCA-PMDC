import numpy as np


def robot_init_2d():
    case = {
        0: {
            'id': 0,
            'robot_type': 'single_int',
            'pos_real': np.array([4.5, 4.5], dtype=np.float32),
            'goal_final': np.array([4, 1], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        },
        1: {
            'id': 1,
            'robot_type': 'single_int',
            'pos_real': np.array([4, 1], dtype=np.float32),
            'goal_final': np.array([-4, 3.0], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        },
        2: {
            'id': 2,
            'robot_type': 'single_int',
            'pos_real': np.array([4.5, 3], dtype=np.float32),
            'goal_final': np.array([-2, -3.5], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        },
        3: {
            'id': 3,
            'robot_type': 'single_int',
            'pos_real': np.array([4.5, 2], dtype=np.float32),
            'goal_final': np.array([3, -1], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        },
        4: {
            'id': 4,
            'robot_type': 'single_int',
            'pos_real': np.array([0, 4.5], dtype=np.float32),
            'goal_final': np.array([4, 2.8], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        },
        5: {
            'id': 5,
            'robot_type': 'single_int',
            'pos_real': np.array([3, -2.5], dtype=np.float32),
            'goal_final': np.array([2.6, 0], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        },
        6: {
            'id': 6,
            'robot_type': 'single_int',
            'pos_real': np.array([0.5, -4], dtype=np.float32),
            'goal_final': np.array([-4, 2], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        },
        7: {
            'id': 7,
            'robot_type': 'single_int',
            'pos_real': np.array([0.0, -1], dtype=np.float32),
            'goal_final': np.array([-2.7, 1.6], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        },
    }


    return case