import numpy as np


def robot_init_2d():
    r = 4.0
    mode = 5
    case1 = {
        i: {
            'id': i,
            'robot_type': 'single_int',
            'pos_real': np.array([r * np.cos(i * 2 * np.pi / mode), r * np.sin(i * 2 * np.pi / mode)],
                                 dtype=np.float32),
            'goal_final': - np.array([r * np.cos(i * 2 * np.pi / mode), r * np.sin(i * 2 * np.pi / mode)],
                                     dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'radius': 0.2,
        } for i in range(mode)
    }
    case2 = {
        mode: {
            'id': mode,
            'robot_type': 'passive',
            'pos_real': np.array([-2, -2], dtype=np.float32),
            'goal_final': np.array([2, 2], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'speed_max': 0.2,
            'radius': 0.2,
        },
        (mode + 1): {
            'id': (mode + 1),
            'robot_type': 'passive',
            'pos_real': np.array([0, -2], dtype=np.float32),
            'goal_final': np.array([0, 2], dtype=np.float32),
            'pos_noise': (0.10 ** 2) * np.eye(2),
            'speed_real': np.zeros(2),
            'speed_max': 0.2,
            'radius': 0.2,
        }
    }

    case = {**case1, **case2}
    return case