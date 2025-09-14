import numpy as np


def check_collision_for_obs(multi_robot_keys: list, multi_robot_pos_real: dict, multi_robot_goal_real: dict, multi_robot_radius: dict,
                    multi_obs_keys: list, multi_obs_state: dict):
    # here we assume that the obstacle is real circle
    # init
    collision_flags = {k: False for k in multi_robot_keys}
    len_keys = len(multi_robot_keys)
    # check robots
    for i in range(len_keys):
        ego_robot_pos_real = multi_robot_pos_real[multi_robot_keys[i]]
        ego_robot_radius = multi_robot_radius[multi_robot_keys[i]]
        for j in range(i + 1, len_keys):
            follower_robot_pos_real = multi_robot_pos_real[multi_robot_keys[j]]
            follower_robot_radius = multi_robot_radius[multi_robot_keys[j]]
            distance_vector = ego_robot_pos_real - follower_robot_pos_real
            threshold = ego_robot_radius + follower_robot_radius
            if np.linalg.norm(distance_vector) < threshold:
                collision_flags[multi_robot_keys[i]] = True
                collision_flags[multi_robot_keys[j]] = True

    # check robot and obs
    for robot_id in multi_robot_keys:
        if collision_flags[robot_id]:
            continue
        ego_robot_pos_real = multi_robot_pos_real[robot_id]
        ego_robot_radius = multi_robot_radius[robot_id]
        for k in multi_obs_keys:
            if multi_obs_state[k]['obs_type'] == 'circle_obs':
                if check_collision_with_robot_and_circle_obs(ego_robot_pos_real, ego_robot_radius,
                                                             multi_obs_state[k]['pos_real'],
                                                             multi_obs_state[k]['radius']):
                    collision_flags[robot_id] = True
                    break
            elif multi_obs_state[k]['obs_type'] == 'rectangle_obs':
                if check_collision_with_robot_and_rectangle_obs(ego_robot_pos_real, ego_robot_radius, multi_obs_state[k]['pos_real'],
                                                                multi_obs_state[k]['size'], multi_obs_state[k]['yaw']):
                    collision_flags[robot_id] = True
                    break

    return collision_flags

def check_collision_with_robot_and_circle_obs(robot_pos_real: np.ndarray, robot_radius, obs_pos_real: np.ndarray, obs_radius):
    distance_vector = np.array(obs_pos_real) - np.array(robot_pos_real)
    threshold = obs_radius + robot_radius
    if np.linalg.norm(distance_vector) < threshold:
        return True
    return False

def check_collision_with_robot_and_rectangle_obs(robot_pos_real: np.ndarray, robot_radius, obs_pos_real: np.ndarray,
                                                 obs_size, obs_yaw):
    distance_vector = np.array(robot_pos_real) - np.array(obs_pos_real)
    distance = np.linalg.norm(distance_vector)
    alpha = np.arctan2(distance_vector[1], distance_vector[0])
    d1 = distance * np.abs(np.cos(alpha - obs_yaw))
    d1 = max(d1 - robot_radius, 0)
    d2 = distance * np.abs(np.sin(alpha - obs_yaw))
    d2 = max(d2 - robot_radius, 0)
    if d1 < obs_size[0] / 2 and d2 < obs_size[1] / 2:
        return True
    return False
                