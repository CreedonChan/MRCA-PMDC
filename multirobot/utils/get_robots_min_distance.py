import numpy as np


def get_robots_min_distance(multi_robot_pos_real: dict[np.ndarray], multi_robot_type: dict):
    min_distance = -1
    keys = []
    for k, v in multi_robot_type.items():
        if v != 'passive':
            keys.append(k)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            distance = np.linalg.norm(multi_robot_pos_real[keys[i]] - multi_robot_pos_real[keys[j]])
            if min_distance > distance or min_distance == -1:
                min_distance = distance

    return min_distance
