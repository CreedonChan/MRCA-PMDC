import time
from datetime import datetime
from statistics import mean

from config.map3 import initialize_system_and_visualizer
# from config.map4 import initialize_system_and_visualizer
# from config.map9 import initialize_system_and_visualizer

from multirobot.utils.get_robot_and_obs_min_distance import get_robot_and_obs_min_distance
from multirobot.utils.get_robots_min_distance import get_robots_min_distance

# simulation_num
sim_num = 1

# data logging
collision_num = 0
success_num = 0
time_out_num = 0

sim_time: dict = {}
running_time: dict = {}
average_running_time: dict = {}

robots_min_distance: dict = {}
robot_and_obs_min_distance: dict = {}

average_travelled_distance: dict = {}

system = None
r_record = 0

for r in range(sim_num):
    current_time = datetime.now()
    # loop preparation
    n_loop = 0
    max_loop = 1000

    # system and visualizer
    if r == r_record:
        system, visualizer = initialize_system_and_visualizer()
        visualizer.show_display = True
    else:
        system, visualizer = initialize_system_and_visualizer(visualizer_init=False)
        # visualizer.show_display = True

    # record the robot arrival
    if_robots_arrived = {k: False for k, _ in system.multi_robot.items()}

    # init the running time list
    r_running_time: list = []

    # init the robot and obs dist
    r_robots_min_distance = -1
    r_robot_and_obs_min_distance = -1

    # main loop
    while n_loop < max_loop:
        # control loop
        n_loop = n_loop + 1

        # get system state
        system.get_system_state()

        # simulate one step
        start_time = time.perf_counter()
        system.simulate_one_step()
        end_time = time.perf_counter()
        r_running_time.append(end_time - start_time)

        # update visualizer
        if r == r_record:
            robot_current_state = {k: {'pos_real': v} for k, v in system.multi_robot_pos_real.items()}
            obs_current_state = system.multi_obs_state
            visualizer.update(robot_current_state, obs_current_state, system.dt * n_loop)

        # distance data store
        robots_distance = get_robots_min_distance(system.multi_robot_pos_real, system.multi_robot_type)
        if r_robots_min_distance > robots_distance or r_robots_min_distance == -1:
            r_robots_min_distance = robots_distance

        multi_obs_pos_real = {k: v['pos_real'] for k, v in system.multi_obs_state.items()}
        robot_and_obs_distance = get_robot_and_obs_min_distance(system.multi_robot_pos_real, multi_obs_pos_real, system.multi_robot_type)
        if r_robot_and_obs_min_distance > robot_and_obs_distance or r_robot_and_obs_min_distance == -1:
            r_robot_and_obs_min_distance = robot_and_obs_distance

        # arrived checking
        for k, v in system.multi_robot.items():
            if_robots_arrived[k] = v.is_arrived

        if list(if_robots_arrived.values()).count(True) == system.n_robot:
            success_num = success_num + 1
            # add r_running_time to running time dict
            running_time[r] = r_running_time
            # add sim_time
            sim_time[r] = n_loop * system.dt
            # add min distance
            robots_min_distance[r] = r_robots_min_distance
            robot_and_obs_min_distance[r] = r_robot_and_obs_min_distance
            # add average travelled distance
            sum_travelled_distance = 0
            for _, v in system.multi_robot.items():
                sum_travelled_distance = sum_travelled_distance + v.travelled_distance
            average_travelled_distance[r] = sum_travelled_distance / system.n_robot
            # add average running time
            average_running_time[r] = mean(running_time[r])
            print("All robots arrived!")
            break

        # collision checking
        system.collision_checking()

        if list(system.collision_flags.values()).count(True) > 0:
            collision_num = collision_num + 1
            print("Collision happens!")
            break

    # time out checking
    if list(if_robots_arrived.values()).count(True) < system.n_robot and n_loop >= max_loop:
        time_out_num = time_out_num + 1
        print('Time out!')