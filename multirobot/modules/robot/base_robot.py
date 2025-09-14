import numpy as np
from numpy.f2py.auxfuncs import throw_error


class BaseRobot:
    dim: int  # dimension of the robot
    id: int  # index of robot

    # radius
    radius: float

    # time step
    dt: float
    horizon: int

    # distance threshold
    collision_threshold: float

    # control max

    # goal
    goal_final: np.array = []
    goal_current: np.array = []
    goal_tolerance = 0.10

    # workplace bounds
    bound_global: np.ndarray = []       # [(x1, x2), (y1, y2)]
    bound_radius: float

    # workplace information
    robots_info_local: dict = {}
    obs_infos_local: dict = {}

    # local robots prediction
    robots_local_predicted_pos: dict = {}

    # real state
    pos_real: np.ndarray = []
    speed_real: np.ndarray = []

    # measured state
    pos_mea: np.ndarray = []
    pos_mea_cov: np.ndarray = []
    speed_mea: np.ndarray = []

    # estimated state
    pos_est: np.ndarray = []
    pos_est_cov: np.ndarray = []
    speed_est: np.ndarray = []
    speed_est_cov: np.ndarray = []

    # control input
    u: np.ndarray = np.zeros(2)
    history_u = [np.zeros(2)]

    # travelled distance
    travelled_distance = 0.0

    # update_num
    update_num = 0

    # deadlock resolving
    is_collision = False
    is_arrived = False
    is_deadlock = False

    # MPPi
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_id(self, robot_id):
        self.id = robot_id

    def set_start_pos(self, start_pos: np.ndarray):
        self.pos_real = start_pos

    def set_goal(self, goal: np.ndarray):
        self.goal_final = goal

    def set_pos_noise(self, cov: np.ndarray):
        self.pos_mea_cov = cov

    def set_init_speed(self, speed: np.ndarray):
        self.speed_real = speed

    def set_radius(self, radius):
        self.radius = radius

    def get_measured_state(self):
        self.pos_mea = self.pos_real.copy()
        # add some noise to the measured state
        noise = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.pos_mea_cov)
        self.pos_mea += noise

        # assume that the measured speed is the real
        self.speed_mea = self.speed_real.copy()

    def get_estimated_state(self):
        self.get_measured_state()
        # let measured state to be the estimated state
        self.pos_est = self.pos_mea.copy()
        self.pos_est_cov = self.pos_mea_cov.copy()

        # assume that we use one special method to est the speed of robot
        self.speed_est, self.speed_est_cov = self.estimate_speed()

    def is_in_local_bound(self, position: np.ndarray):
        distance = np.linalg.norm(self.pos_est - position)
        if distance < self.bound_radius:
            return True
        return False

    def get_local_robots_info(self, multi_robot_pos_est: dict, multi_robot_pos_est_cov: dict, multi_robot_radius: dict,
                              multi_robot_speed_est: dict, multi_robot_speed_est_cov: dict, multi_robot_type: dict):
        # init the local robots info
        self.robots_info_local = {}
        for k, v in multi_robot_pos_est.items():
            if k == self.id:
                continue
            if self.is_in_local_bound(multi_robot_pos_est[k]):
                self.robots_info_local[k] = {}
                self.robots_info_local[k]['pos_est'] = multi_robot_pos_est[k]
                self.robots_info_local[k]['pos_est_cov'] = multi_robot_pos_est_cov[k]
                self.robots_info_local[k]['radius'] = multi_robot_radius[k]
                self.robots_info_local[k]['speed_est'] = multi_robot_speed_est[k]
                self.robots_info_local[k]['speed_est_cov'] = multi_robot_speed_est_cov[k]
                self.robots_info_local[k]['robot_type'] = multi_robot_type[k]

    def get_local_obs_info(self, multi_obs_state: dict):
        """It only considers the circle shape of the obstacle"""
        # init the local obs info
        self.obs_infos_local = {}
        for k, v in multi_obs_state.items():
            if self.is_in_local_bound(v['pos_est']):
                if v['obs_type'] == 'circle_obs':
                        self.obs_infos_local[k] = {}
                        self.obs_infos_local[k]['pos_est'] = v['pos_est']
                        self.obs_infos_local[k]['radius'] = v['radius']
                        self.obs_infos_local[k]['obs_type'] = v['obs_type']
                elif v['obs_type'] == 'rectangle_obs':
                        self.obs_infos_local[k] = {}
                        self.obs_infos_local[k]['pos_est'] = v['pos_est']
                        self.obs_infos_local[k]['pos_est_cov'] = v['pos_est_cov']
                        self.obs_infos_local[k]['size'] = v['size']
                        self.obs_infos_local[k]['yaw'] = v['yaw']
                        self.obs_infos_local[k]['obs_type'] = v['obs_type']
                else:
                    throw_error('Obstacle type not defined!')

    def predict_local_robots(self):
        # use the simple predicted method
        for k, v in self.robots_info_local.items():
            self.robots_local_predicted_pos[k] =  v['pos_est'] + v['speed_est'] * self.dt

    def estimate_speed(self):
        # here we assume robot calculate the estimation of the speed by themselves
        to_goal_vector = self.goal_final - self.pos_est
        distance_to_final_goal = np.linalg.norm(to_goal_vector)
        to_goal_normal_vector = to_goal_vector / distance_to_final_goal
        if self.is_arrived:
            vel_speed = 0.0
        else:
            vel_speed = min(self.speed_max, distance_to_final_goal / self.dt)
        speed_est = to_goal_normal_vector * vel_speed

        # here we assume the cov of speed
        speed_est_cov = (0.01 ** 2) * np.eye(self.dim)

        return speed_est, speed_est_cov

    def calculate_control_input(self):
        pass

    def simulate_one_step(self):
        pass

    def state_checking(self):
        # initial
        self.is_deadlock = False
        # check if arrived at final goal
        distance_to_final_goal = np.linalg.norm(self.pos_real - self.goal_final)
        if distance_to_final_goal < self.goal_tolerance:
            self.is_arrived = True

        # check if in deadlock
        # not reaching the final goal but the velocity is zero
        if not self.is_arrived and np.linalg.norm(self.u) < 1e-3 and len(self.robots_info_local) > 0:
            self.is_deadlock = True
