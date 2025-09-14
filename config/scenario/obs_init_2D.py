import numpy as np


def circle_obx_init_2d(mode: int):
    # case = {
    #     0: {
    #         'n_obs': 0,
    #         'pos_real': None,
    #         'radius': None
    #     },
    #     1: {
    #         'n_obs': 1,
    #         'pos_real': [
    #             np.array([-4, 1])
    #         ],
    #         'radius': [
    #             1.0
    #         ]
    #     },
    #     3: {
    #         'n_obs': 3,
    #         'pos_real': [
    #             np.array([-4.0, 1.0]),
    #             np.array([3.0, 4.0]),
    #             np.array([0.0, -2.0])
    #         ],
    #         'radius': [
    #             0.5,
    #             0.5,
    #             0.5
    #         ]
    #     }
    # }

    case = {
        0: {},
        1: {
            0: {
                'id': 0,
                'obs_type': 'circle_obs',
                'pos_real': np.array([0.0, 0.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'radius': 1.0,
            }
        },
        3: {
            0: {
                'id': 0,
                'obs_type': 'circle_obs',
                'pos_real': np.array([-4.0, 1.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'radius': 0.5,
            },
            1: {
                'id': 1,
                'obs_type': 'circle_obs',
                'pos_real': np.array([3.0, 4.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'radius': 0.5,
            },
            2: {
                'id': 2,
                'obs_type': 'circle_obs',
                'pos_real': np.array([0.0, -2.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'radius': 0.5,
            }
        }
    }
    return case[mode]

def rectangle_obs_init_2d(mode: int):
    case = {
        0: {},
        # 1: {
        #     0: {
        #         'id': 0,
        #         'obs_type': 'rectangle_obs',
        #         'pos_real': np.array([-4, 1]),
        #         'pos_noise': (0.10 ** 2) * np.eye(2),
        #         'size': np.array([3, 2]),
        #         'yaw': 0.0,
        #     }
        # },
        1: {
            0: {
                'id': 0,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([0.0, 0.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([2, 2]),
                'yaw': 0.0,
            }
        },
        2: {
            0: {
                'id': 0,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([0.5, 1.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            1: {
                'id': 1,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-0.5, -1.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            }
        },
        3: {
            0: {
                'id': 0,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-4.0, 1.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            1: {
                'id': 1,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([3.0, 4.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            2: {
                'id': 2,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([0.0, -2.0]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
        },
        10: {
            0: {
                'id': 0,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-3, 4.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            1: {
                'id': 1,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-2, 3.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            2: {
                'id': 2,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-4.5, 1]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            3: {
                'id': 3,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-1, -4.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            4: {
                'id': 4,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-2.5, -1.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            5: {
                'id': 5,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([1.5, -2.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            6: {
                'id': 6,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([4.5, -0.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            7: {
                'id': 7,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([2.5, 3.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            8: {
                'id': 8,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([0.5, 0.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            9: {
                'id': 9,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-4.5, -4.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
        },
        11: {
            0: {
                'id': 0,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-3, 4.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            1: {
                'id': 1,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-2, 3.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            2: {
                'id': 2,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-4.5, 1]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            3: {
                'id': 3,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-1, -4.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            4: {
                'id': 4,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-2.5, -1.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            5: {
                'id': 5,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([1.5, -2.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            6: {
                'id': 6,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([4.5, -0.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            7: {
                'id': 7,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([3.0, 3.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            8: {
                'id': 8,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([0.5, 0.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            9: {
                'id': 9,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([-4.5, -4.5]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
            10: {
                'id': 10,
                'obs_type': 'rectangle_obs',
                'pos_real': np.array([0.5, 2.9]),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'size': np.array([1, 1]),
                'yaw': 0.0,
            },
        }
    }

    return case[mode]

