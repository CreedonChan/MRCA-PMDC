import numpy as np


def robot_init_2d(mode: int):
    # # case 0
    r = 5.0
    # angle = 2 * np.pi / 5
    #
    # # case 6
    # angle6 = 2 * np.pi / 6
    #
    # # case 7
    # angle7 = 2 * np.pi / 7

    # case ={
    #     0: {
    #         'n_robot': 5,
    #         'pos_real': [
    #             np.array([r * np.cos(0 * angle), r * np.sin(0 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(1 * angle), r * np.sin(1 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(2 * angle), r * np.sin(2 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(3 * angle), r * np.sin(3 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(4 * angle), r * np.sin(4 * angle)], dtype=np.float32),
    #         ],
    #         'radius': [
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2,
    #         ]
    #     },
    #     1: {
    #         'n_robot': 1,
    #         'pos_real': [
    #             np.array([r * np.cos(0 * angle), r * np.sin(0 * angle)], dtype=np.float32),
    #         ],
    #         'radius': [
    #             0.2,
    #         ]
    #     },
    #     2: {
    #         'n_robot': 2,
    #         'pos_real': [
    #             np.array([r * np.cos(0 * angle), r * np.sin(0 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(1 * angle), r * np.sin(1 * angle)], dtype=np.float32),
    #         ],
    #         'radius': [
    #             0.2,
    #             0.2,
    #         ]
    #     },
    #     3: {
    #         'n_robot': 3,
    #         'pos_real': [
    #             np.array([r * np.cos(0 * angle), r * np.sin(0 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(1 * angle), r * np.sin(1 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(3 * angle), r * np.sin(3 * angle)], dtype=np.float32),
    #         ],
    #         'radius': [
    #             0.2,
    #             0.2,
    #             0.2
    #         ]
    #     },
    #     4: {
    #         'n_robot': 4,
    #         'pos_real': [
    #             np.array([r * np.cos(0 * angle), r * np.sin(0 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(1 * angle), r * np.sin(1 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(3 * angle), r * np.sin(3 * angle)], dtype=np.float32),
    #             np.array([r * np.cos(4 * angle), r * np.sin(4 * angle)], dtype=np.float32),
    #         ],
    #         'radius': [
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2
    #         ]
    #     },
    #     6: {
    #         'n_robot': 6,
    #         'pos_real': [
    #             np.array([r * np.cos(0 * angle6), r * np.sin(0 * angle6)], dtype=np.float32),
    #             np.array([r * np.cos(1 * angle6), r * np.sin(1 * angle6)], dtype=np.float32),
    #             np.array([r * np.cos(2 * angle6), r * np.sin(2 * angle6)], dtype=np.float32),
    #             np.array([r * np.cos(3 * angle6), r * np.sin(3 * angle6)], dtype=np.float32),
    #             np.array([r * np.cos(4 * angle6), r * np.sin(4 * angle6)], dtype=np.float32),
    #             np.array([r * np.cos(5 * angle6), r * np.sin(5 * angle6)], dtype=np.float32),
    #         ],
    #         'radius': [
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2
    #         ]
    #     },
    #     7: {
    #         'n_robot': 7,
    #         'pos_real': [
    #             np.array([r * np.cos(0 * angle7), r * np.sin(0 * angle7)], dtype=np.float32),
    #             np.array([r * np.cos(1 * angle7), r * np.sin(1 * angle7)], dtype=np.float32),
    #             np.array([r * np.cos(2 * angle7), r * np.sin(2 * angle7)], dtype=np.float32),
    #             np.array([r * np.cos(3 * angle7), r * np.sin(3 * angle7)], dtype=np.float32),
    #             np.array([r * np.cos(4 * angle7), r * np.sin(4 * angle7)], dtype=np.float32),
    #             np.array([r * np.cos(5 * angle7), r * np.sin(5 * angle7)], dtype=np.float32),
    #             np.array([r * np.cos(6 * angle7), r * np.sin(6 * angle7)], dtype=np.float32),
    #         ],
    #         'radius': [
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2,
    #             0.2
    #         ]
    #     }
    # }

    # case = {
    #     mode: {
    #         'n_robot': mode,
    #         'pos_real': [
    #             np.array([r * np.cos(i * 2 * np.pi / mode), r * np.sin(i * 2 * np.pi / mode)], dtype=np.float32)
    #             for i in range(mode)
    #         ],
    #         'radius': [
    #             0.2
    #         ] * mode
    #     },
    # }

    case = {
        mode: {
            i: {
                'id': i,
                'robot_type': 'single_int',
                'pos_real': np.array([r * np.cos(i * 2 * np.pi / mode), r * np.sin(i * 2 * np.pi / mode)], dtype=np.float32),
                'goal_final': - np.array([r * np.cos(i * 2 * np.pi / mode), r * np.sin(i * 2 * np.pi / mode)], dtype=np.float32),
                'pos_noise': (0.10 ** 2) * np.eye(2),
                'speed_real': np.zeros(2),
                'radius': 0.2,
            } for i in range(mode)
        },
    }
    return case[mode]