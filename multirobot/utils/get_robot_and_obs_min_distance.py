import numpy as np


def get_robot_and_obs_min_distance(multi_robot_pos_real: dict, multi_obs_pos_real: dict, multi_robot_type: dict):
    min_distance = -1
    keys = []
    for k, v in multi_robot_type.items():
        if v == 'passive':
            keys.append(k)
    for k1, v1 in multi_robot_pos_real.items():
        if multi_robot_type[k1] == 'passive':
            continue
        for _, v2 in multi_obs_pos_real.items():
            distance = np.linalg.norm(np.array(v1) - np.array(v2))
            if min_distance > distance or min_distance == -1:
                min_distance = distance
        for k in keys:
            distance = np.linalg.norm(np.array(v1) - np.array(multi_robot_pos_real[k]))
            if min_distance > distance or min_distance == -1:
                min_distance = distance

    return min_distance