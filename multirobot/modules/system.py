import numpy as np
from numpy.f2py.auxfuncs import throw_error

from multirobot.modules.obstacle.base_obs import BaseObs
from multirobot.modules.obstacle.circle_obs import CircleObs
from multirobot.modules.obstacle.rectangle_obs import RectangleObs
from multirobot.modules.robot.base_robot import BaseRobot
from multirobot.modules.robot.passive_robot import PassiveRobot
from multirobot.modules.robot.single_int_robot import SingleIntRobot
from multirobot.utils.check_collision import check_collision_for_obs


class System:
    dim: int
    dt: float
    ws: np.ndarray

    # time
    time_global: int = 0

    # object
    n_robot: int
    n_obs: int
    multi_robot: dict = {}
    multi_obs: dict = {}
    multi_robot_keys: list = []
    multi_obs_keys: list = []

    # system current robots real state
    multi_robot_goal_real: dict = {}
    multi_robot_pos_real: dict = {}
    multi_robot_speed_real: dict = {}

    # system current robots est state
    multi_robot_pos_est: dict = {}
    multi_robot_pos_est_cov: dict = {}
    multi_robot_speed_est: dict = {}
    multi_robot_speed_est_cov: dict = {}

    # system current robots radius
    multi_robot_radius: dict = {}

    # system current obs state
    multi_obs_state: dict = {}

    # system current robots type
    multi_robot_type: dict = {}

    # collision info
    collision_flags = {}    # here had not been initialized

    def __init__(self, robot_database: dict, obs_database: dict, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        # init the num and keys of the robots and obs
        self.multi_robot_keys = list(robot_database.get('robots_config').keys())
        self.multi_obs_keys = list(obs_database.get('obs_config').keys())
        self.n_robot = len(self.multi_robot_keys)
        self.n_obs = len(self.multi_obs_keys)

        # init the robot and the obs
        for i in self.multi_robot_keys:
            if robot_database['robots_config'][i]['robot_type'] == 'single_int':
                self.multi_robot[i] = SingleIntRobot(robot_database.get('robots_config')[i], **robot_database.get('basic_config'))
                self.multi_robot_type[i] = 'single_int'
                self.multi_robot_radius[i] = robot_database.get('robots_config')[i]['radius']
            elif robot_database['robots_config'][i]['robot_type'] == 'passive':
                self.multi_robot[i] = PassiveRobot(robot_database.get('robots_config')[i], **robot_database.get('basic_config'))
                self.multi_robot_type[i] = 'passive'
                self.multi_robot_radius[i] = robot_database.get('robots_config')[i]['radius']
            else:
                throw_error('Robot type not defined!')

        for i in self.multi_obs_keys:
            # here can consider other type of the obstacle
            # init the obs_state
            self.multi_obs_state[i] = {}
            if obs_database['obs_config'][i]['obs_type'] == 'circle_obs':
                self.multi_obs[i] = CircleObs(obs_database.get('obs_config')[i], **obs_database.get('basic_config'))
                self.multi_obs_state[i]['obs_type'] = 'circle_obs'
                self.multi_obs_state[i]['radius'] = obs_database.get('obs_config')[i]['radius']
            elif obs_database['obs_config'][i]['obs_type'] == 'rectangle_obs':
                self.multi_obs[i] = RectangleObs(obs_database.get('obs_config')[i], **obs_database.get('basic_config'))
                self.multi_obs_state[i]['obs_type'] = 'rectangle_obs'
                self.multi_obs_state[i]['size'] = obs_database.get('obs_config')[i]['size']
                self.multi_obs_state[i]['yaw'] = obs_database.get('obs_config')[i]['yaw']
            else:
                throw_error('Obstacle type not defined!')

        # init the pos of all body
        self.get_system_state()

        # initialize collision info
        self.collision_flags = {k: False for k, _ in self.multi_robot.items()}

    def get_system_state(self):
        for k, v in self.multi_robot.items():
            # real
            self.multi_robot_pos_real[k] = v.pos_real
            self.multi_robot_goal_real[k] = v.goal_final
            self.multi_robot_speed_real[k] = v.speed_real
            # est
            v.get_estimated_state()
            self.multi_robot_pos_est[k] = v.pos_est
            self.multi_robot_pos_est_cov[k] = v.pos_est_cov
            self.multi_robot_speed_est[k] = v.speed_est
            self.multi_robot_speed_est_cov[k] = v.speed_est_cov

        for k, v in self.multi_obs.items():
            # real
            self.multi_obs_state[k]['pos_real'] = v.pos_real
            # est
            v.get_estimated_state()
            self.multi_obs_state[k]['pos_est'] = v.pos_est
            self.multi_obs_state[k]['pos_est_cov'] = v.pos_est_cov

    def simulate_one_step(self):
        # time
        self.time_global = self.time_global + 1

        # each robot
        for k, v in self.multi_robot.items():
            # # ego robot estimated state
            # v.pos_est = self.multi_robot_pos_est[k]
            # v.pos_est_cov = self.multi_robot_pos_est_cov[k]

            # obtain local robot info
            v.get_local_robots_info(self.multi_robot_pos_est, self.multi_robot_pos_est_cov, self.multi_robot_radius, self.multi_robot_speed_est, self.multi_robot_speed_est_cov, self.multi_robot_type)

            # obtain local obs info
            v.get_local_obs_info(self.multi_obs_state)

            # # predict local robot pos
            # v.predict_local_robots()

            # state checking
            v.state_checking()

            # calculate control input
            v.calculate_control_input()

            # arrive checking
            if v.is_arrived:
                v.u = - 0.0 * v.u

            # collision checking
            if v.is_collision:
                v.u = - 0.0 * v.u
                print(f'Robot ID:{k} is in collision!')

            # simulate the robot one step
            v.simulate_one_step()

    def collision_checking(self):
        # here we assume that the obstacle is real circle
        self.collision_flags = check_collision_for_obs(self.multi_robot_keys, self.multi_robot_pos_real,
                                                       self.multi_robot_goal_real, self.multi_robot_radius,
                                                       self.multi_obs_keys, self.multi_obs_state)

        # transmit to objective
        for k, v in self.multi_robot.items():
            if not v.is_collision:
                v.is_collision = self.collision_flags[k]

    @ property
    def multi_robot_controls(self):
        return self.multi_robot_speed_real

    def locate_robot_and_obs_positions(self, robot_positions, obs_positions):
        for k, v in self.multi_robot.items():
            # real
            v.pos_real = np.array(robot_positions[k])
            self.multi_robot_pos_real[k] = v.pos_real
            # est
            v.get_estimated_state()
            self.multi_robot_pos_est[k] = v.pos_est
            self.multi_robot_pos_est_cov[k] = v.pos_est_cov

        for k, v in self.multi_obs.items():
            # real
            v.pos_real = np.array(obs_positions[k])
            self.multi_obs_state[k]['pos_real'] = v.pos_real
            # est
            v.get_estimated_state()
            self.multi_obs_state[k]['pos_est'] = v.pos_est
            self.multi_obs_state[k]['pos_est_cov'] = v.pos_est_cov

