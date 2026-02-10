import time
from copy import copy

import torch
import numpy as np
from mpmath.math2 import sqrt2
from numpy.f2py.auxfuncs import throw_error

from multirobot.control_method.mppi import MPPIController
from multirobot.modules.robot.base_robot import BaseRobot
from multirobot.utils.mathematic import magnitude_angle_to_xy, angle_between_2d_vectors, gershgorin_lower_bound
from torch.distributions import Normal


class SingleIntRobot(BaseRobot):
    # control max
    speed_max = 0.0

    def __init__(self, init_data: dict, **kwargs):
        super().__init__(**kwargs)

        # robot type
        self.robot_type = 'single_int'

        # init the data (here you can initialize some parameter with no definition)
        self.set_id(init_data.get('id'))
        self.set_start_pos(init_data.get('pos_real'))
        self.set_pos_noise(init_data.get('pos_noise'))
        self.set_goal(init_data.get('goal_final'))
        self.set_init_speed(init_data.get('speed_real', np.zeros(self.dim)))
        self.set_radius(init_data.get('radius'))

        # control max
        self.speed_max: float = init_data.get('speed_max', kwargs.get('speed_max'))

        # control input
        self.u = np.zeros(2, dtype=np.float32)

        # MPPi
        self.mppi: MPPIController = None
        # init in MPPi
        self.n_sample = 1000
        self.horizon = 10
        # weight in MPPi
        self.w_state = 0.0
        self.w_action = 0.0
        self.w_terminal_state = 200.0
        self.w_smooth = 0.0
        self.w_speed = 0.0
        self.w_control_to_goal = 0.0
        self.w_dis_to_obs = 1.2
        self.Q = (0.06**2) * torch.eye(self.dim)

        # define F and noise
        self.F = torch.tensor([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=torch.float32)

        self.q_pos = 0.01 ** 2
        self.q_speed = 0.001 ** 2
        self.robot_noises = torch.block_diag(self.q_pos * torch.eye(self.dim), self.q_speed * torch.eye(self.dim))

        # for the constant velocity model
        self.sigma0 = None

        # tensor data preparation
        # normal ppf
        self.std_normal = Normal(loc=0, scale=1)
        # goal cost
        self.goal_tensor = torch.from_numpy(self.goal_final)
        self.pos_est_tensor = None
        self.pos_est_cov_tensor = None
        self._len_to_goal = None
        self.len_to_goal = None

        # robot collision check
        self.robot_pos_ests: torch.Tensor = None
        self.robot_speed_ests: torch.Tensor = None
        self.robot_radiuses: torch.Tensor = None
        self.robot_pos_est_covs: torch.Tensor = None
        self.robot_speed_est_covs: torch.Tensor = None

        self._initialize_mppi()

        # boundary checks
        self.bound_lowers = torch.tensor([x[0] for x in self.bound_global])
        self.bound_uppers = torch.tensor([x[1] for x in self.bound_global])

    def simulate_one_step(self):
        self.pos_real = self.pos_real + self.u * self.dt
        self.speed_real = self.u
        # here store the control
        # self.history_u.append(copy(self.u))
        # self.history_u = self.history_u[:]
        # this is for the travelled distance
        self.travelled_distance = self.travelled_distance + np.linalg.norm(self.u * self.dt)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor):
        next_state = state + action * self.dt
        return next_state

    def generate_running_cost(self, state_num: int=None):
        # store the action last
        last_action = torch.tensor(self.u).expand(state_num, -1)
        stage_count = -1

        def running_cost(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            """
            Compute the running cost for MPPI controller considering:
            - Goal distance cost
            - Collision cost with other robots and obstacles
            - Boundary violation cost
            - Control effort cost

            Args:
                state: Current state tensor of shape (B, state_dim)
                action: Control action tensor of shape (B, action_dim)

            Returns:
                Total cost tensor of shape (B, *)
            """
            # update the n stage
            nonlocal stage_count
            stage_count = (stage_count + 1) % self.horizon
            curr_stage = stage_count + 1

            # Ensure consistent device and dtype
            device = state.device
            dtype = state.dtype

            # --- Goal Cost --
            # Normalized squared distance to goal
            # goal_cost = self.w_state * torch.sum((self.goal_tensor - state) ** 2, dim=-1) / self.len_to_goal  # (B,)

            # --- Collision Cost ---
            distance_cost = torch.zeros(state_num)

            # --- Robot Collision Cost (Vectorized) ---
            if self.robots_info_local:
                # Calculate predicted positions with noise
                # noise = self.noise_samples[:curr_stage].sum(dim=0).to(dtype=dtype, device=device)
                other_robot_pos = (
                        self.robot_pos_ests.to(dtype=dtype, device=device) +
                        curr_stage * self.dt * self.robot_speed_ests.to(dtype=dtype, device=device)
                )

                # Calculate combined covariance
                self.sigma0 = torch.matmul(self.F, self.sigma0)
                self.sigma0 = torch.matmul(self.sigma0, self.F.T)
                self.sigma0 += self.robot_noises
                robot_pos_est_covs = self.sigma0[:, :self.dim, :self.dim]

                cov = (
                    robot_pos_est_covs +
                    self.pos_est_cov_tensor.to(dtype=dtype, device=device) +
                    curr_stage * self.Q.to(dtype=dtype, device=device)
                )

                # Vectorized Mahalanobis distance calculation
                delta = state.unsqueeze(1) - other_robot_pos.unsqueeze(0)  # Shape: (B, N, 2)
                cov_inv = torch.linalg.inv(cov)
                weighted_dist = torch.einsum('bni,nij,bnj->bn', delta, cov_inv, delta)
                # det = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]
                # inv_cov = torch.stack([
                #     torch.stack([cov[..., 1, 1], -cov[..., 0, 1]], dim=-1),
                #     torch.stack([-cov[..., 1, 0], cov[..., 0, 0]], dim=-1)
                # ], dim=-2) / det.unsqueeze(-1).unsqueeze(-1)

                # weighted_dist = torch.einsum('bni,nij,bnj->bn', delta, inv_cov, delta)

                # Vectorized threshold calculation
                eigs = torch.linalg.eigvalsh(cov)
                min_eig = eigs.min(dim=1).values
                min_eig_sq = torch.sqrt(min_eig)
                safe_ratio = (self.radius + self.robot_radiuses)/ min_eig_sq
                # threshold = (
                #     (safe_distance / min_eig_sq) - self.std_normal.icdf(0.5 * self.collision_threshold / (self.std_normal.cdf(safe_distance / min_eig_sq) - 0.5))
                # ) ** 2

                # min_eig_lower_bound = gershgorin_lower_bound(cov)
                # min_eig_lower_bound_eq = torch.sqrt(min_eig_lower_bound)
                # safe_ratio = (self.radius + self.robot_radiuses) / min_eig_lower_bound_eq

                threshold = (safe_ratio - sqrt2 * torch.erfinv((2 * self.collision_threshold / (torch.erf(safe_ratio / sqrt2))) - 1)) ** 2

                # Apply penalty to any violating pairs
                collision_mask = torch.any(weighted_dist < threshold.unsqueeze(0), dim=1)
                distance_cost.masked_fill_(collision_mask, 1e6)

            # Obstacle collision checks
            # here we consider the obstacle uncertainty
            for obs_id, obs_info in self.obs_infos_local.items():
                if obs_info['obs_type'] == 'circle_obs':
                    obs_pos = torch.tensor(obs_info['pos_est'], device=device, dtype=dtype)
                    dist = torch.norm(state - obs_pos, dim=1)
                    distance_cost.masked_fill_(dist < (self.radius + self.w_dis_to_obs * obs_info['radius']), 1e6)
                elif obs_info['obs_type'] == 'rectangle_obs':
                    # # Cal the pos and dist of robot and obs
                    # obs_pos = torch.tensor(obs_info['pos_est'], device=device, dtype=dtype)
                    # dist = torch.sum((state - obs_pos)**2, dim=-1)
                    #
                    # # Calculate combined covariance
                    # cov = self.pos_est_cov + obs_info['pos_est_cov']
                    # eigs = np.linalg.eigvals(cov)
                    # max_eig = max(eigs)
                    # min_eig = min(eigs)
                    #
                    # threshold = (np.sqrt(2 * max_eig) * abs(
                    #         self.radius / np.sqrt(min_eig) -
                    #         self.std_normal.icdf(torch.sqrt(torch.tensor(self.collision_threshold, device=device)))
                    # ) + np.linalg.norm(obs_info['size']) / 2) ** 2
                    # mask = dist < threshold
                    # distance_cost.masked_fill_(mask, 1e6)

                    # this is rectangle bigger than real one

                    # # # Cal the pos and dist of robot and obs
                    # obs_pos = torch.tensor(obs_info['pos_est'], device=device, dtype=dtype)
                    # dist_vector = state - obs_pos
                    # dist = torch.sum(dist_vector ** 2, dim=-1)
                    #
                    # # Calculate combined covariance
                    # cov = self.pos_est_cov + obs_info['pos_est_cov']
                    # eigs = np.linalg.eigvals(cov)
                    # max_eig = max(eigs)
                    # min_eig = min(eigs)
                    #
                    # alpha = torch.atan2(dist_vector[:, 1], dist_vector[:, 0])
                    # d1 = dist * torch.abs(torch.cos(alpha - obs_info['yaw']))
                    # d1 = torch.clamp(d1 - self.radius, min=0.0)
                    # d2 = dist * torch.abs(torch.sin(alpha - obs_info['yaw']))
                    # d2 = torch.clamp(d2 - self.radius, min=0.0)
                    #
                    # uncertainty_radius = np.sqrt(2 * max_eig) * abs(
                    #         self.radius / np.sqrt(min_eig) -
                    #         self.std_normal.icdf(torch.sqrt(torch.tensor(self.collision_threshold, device=device)))
                    # )
                    #
                    # mask = (d1 < uncertainty_radius + obs_info['size'][0] / 2) & (d2 < uncertainty_radius + obs_info['size'][1] / 2)
                    # distance_cost.masked_fill_(mask, 1e6)

                    # The third method
                    obs_pos = torch.tensor(obs_info['pos_est'], device=device, dtype=dtype)
                    delta = state - obs_pos
                    # Calculate combined covariance
                    cov = self.pos_est_cov_tensor + curr_stage * self.Q.to(device=device, dtype=dtype) + torch.tensor(obs_info['pos_est_cov'], device=device, dtype=dtype)
                    eigs = np.linalg.eigvals(cov)
                    min_eig = torch.tensor(min(eigs))
                    min_eig_sq = torch.sqrt(min_eig)

                    cov_inv = torch.linalg.inv(cov)
                    weighted_dist = torch.einsum('bi,ij,bj->b', delta, cov_inv, delta)
                    safe_ratio = (self.radius + np.linalg.norm(obs_info['size']) / 2) / min_eig_sq

                    # threshold = (
                    #     (safe_distance / min_eig_sq) - self.std_normal.icdf(0.5 * self.collision_threshold / (self.std_normal.cdf(safe_distance / min_eig_sq) - 0.5))
                    # ) ** 2

                    threshold = (safe_ratio - sqrt2 * torch.erfinv((2 * self.collision_threshold / (torch.erf(safe_ratio / sqrt2))) - 1)) ** 2
                    mask = weighted_dist < threshold
                    distance_cost.masked_fill_(mask, 1e6)

            # --- Boundary Cost (Vectorized) ---
            violates_lower = torch.any(state < self.bound_lowers.to(device), dim=1)
            violates_upper = torch.any(state > self.bound_uppers.to(device), dim=1)
            distance_cost.masked_fill_(violates_lower | violates_upper, 1e6)

            # # --- Control Cost ---
            # action_cost = torch.zeros(state_num)
            # if self.dim == 2:
            #     action_norm = torch.norm(action, p=2, dim=-1)
            #     action_cost = self.w_action * torch.sum(action_norm - self.speed_max)
            #     # mask = action_norm > self.speed_max
            #     # action_cost = torch.where(mask, self.w_action * (action_norm - self.speed_max), 0)
            #
            #     # action_cost = self.w_action * torch.sum(action ** 2, dim=-1) / (self.speed_max ** 2)
            # else:
            #     throw_error('We only have the 2D robot!')

            # ---Speed Penalty ---
            # distance_to_target = torch.sum((goal_tensor - state) ** 2, dim=-1)
            # speed_penalty = self.w_speed * torch.sum(action ** 2, dim=-1) * (1.0 / (0 .1 + distance_to_target))
            # goal_direction_vector = self.speed_max * (self.goal_tensor - state) / torch.norm(self.goal_tensor - state, p=2, dim=-1).reshape(-1, 1)
            # speed_penalty = self.w_speed * torch.sum((action - goal_direction_vector) ** 2, dim=-1)
            # speed_penalty = self.w_speed * (self.speed_max ** 2 - torch.sum(action ** 2, dim=-1)) / (self.speed_max ** 2)

            # --- Control Smooth ---
            # nonlocal last_action
            # delta_direction = angle_between_2d_vectors(action, last_action)
            # last_action = action.clone()
            # smooth_penalty = self.w_smooth * torch.sum(delta_direction ** 2, dim=-1)

            # # --- Control to goal direction ---
            # goal_direction_normal_vector = (goal_tensor - state) / torch.norm(goal_tensor - state, p=2, dim=-1).reshape(-1, 1)
            # control_normal_vector = action / torch.norm(action, p=2, dim=-1).reshape(-1, 1)
            # delta = goal_direction_normal_vector - control_normal_vector
            # control_to_goal_penalty = self.w_control_to_goal * torch.sum(delta ** 2, dim=-1)

            # --- Total Cost ---
            # return self.w_state * goal_cost + distance_cost + action_cost + smooth_penalty + speed_penalty
            return distance_cost
        return running_cost

    def generate_terminal_cost(self, state_num: int):
        def terminal_cost(state, action=None):
            # We put the first stage of the constraint for robots here

            # --- Terminal State Cost ---
            final_state = state[..., -1, :][0]
            len_to_goal = torch.sum((torch.tensor(self.goal_final - self.pos_est)) ** 2)
            len_to_goal = torch.clamp(len_to_goal, min=1.0)  # Avoid division by zero
            terminal_state_cost = self.w_terminal_state * torch.sum((torch.tensor(self.goal_final) - final_state) ** 2, dim=-1) / len_to_goal
            return terminal_state_cost
        return terminal_cost

    def _initialize_mppi(self):
        # initialize the MPPi
        if self.dim == 2:
            self.mppi = MPPIController(dynamics=self.dynamics,
                                       running_cost=self.generate_running_cost(self.n_sample),
                                       nx=self.dim,
                                       noise_sigma=1.0 * self.speed_max * torch.eye(self.dim),
                                       terminal_state_cost=self.generate_terminal_cost(self.n_sample),
                                       lambda_=1.0,
                                       num_samples=self.n_sample,
                                       horizon=self.horizon,
                                       # u_min=-torch.tensor([self.speed_max, self.speed_max]),
                                       # u_max=torch.tensor([self.speed_max, self.speed_max]),
                                       max_action_norm=self.speed_max,)
        else:
            throw_error('We only have the 2D robot!')

    def prepare_tensor_data_for_mppi(self):
        # normal ppf

        # goal cost data
        self.pos_est_tensor = torch.from_numpy(self.pos_est)
        self.pos_est_cov_tensor = torch.tensor(self.pos_est_cov, dtype=torch.float32)
        self._len_to_goal = torch.sum((self.goal_tensor - self.pos_est_tensor) ** 2)
        self.len_to_goal = self._len_to_goal.clamp(min=1.0)

        # Pre-process robot info (convert NumPy/float to tensors once)
        robot_data = []
        for robot_info in self.robots_info_local.values():
            robot_data.append({
                'pos_est': torch.as_tensor(robot_info['pos_est'], dtype=torch.float32),
                'speed_est': torch.as_tensor(robot_info['speed_est'], dtype=torch.float32),
                'radius': float(robot_info['radius']),
                'pos_est_cov': torch.as_tensor(robot_info['pos_est_cov'], dtype=torch.float32),
                'speed_est_cov': torch.as_tensor(robot_info['speed_est_cov'], dtype=torch.float32),
            })

        # Batch robot data if any robots exist
        if robot_data:
            self.robot_pos_ests = torch.stack([d['pos_est'] for d in robot_data])
            self.robot_speed_ests = torch.stack([d['speed_est'] for d in robot_data])
            self.robot_radiuses = torch.tensor([d['radius'] for d in robot_data])
            self.robot_pos_est_covs = torch.stack([d['pos_est_cov'] for d in robot_data])
            self.robot_speed_est_covs = torch.stack([d['speed_est_cov'] for d in robot_data])

            # prepare for prediction
            N = len(robot_data)
            self.sigma0 = torch.zeros(N, 2 * self.dim, 2 * self.dim, device=self.robot_pos_est_covs.device, dtype=self.robot_pos_est_covs.dtype)
            self.sigma0[:, :self.dim, :self.dim] = self.robot_pos_est_covs
            self.sigma0[:, self.dim:, self.dim:] = self.robot_speed_est_covs

        # prepare noise
        # self.noise_samples = torch.randn(self.horizon, len(self.robots_info_local), 2) * torch.sqrt(self.Q[0, 0])

        # Pre-compute boundary checks

    def calculate_control_input(self):
        # pre data
        self.prepare_tensor_data_for_mppi()

        # limit the control and state
        action = self.mppi.command(self.pos_est_tensor)
        next_state = self.dynamics(self.pos_est_tensor, action)
        # here the next_state must be (1, *)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        next_cost = self.generate_running_cost(1)(next_state, action)
        # if arrived, don't move
        if self.is_arrived:
            self.u = - 0.0 * self.u
        # if stuck in deadlock, add some noise to let it out
        # elif self.is_deadlock:
        #     noise = np.random.normal(loc=0, scale=0.1 * self.speed_max, size=self.u.shape)
        #     self.u = self.u + noise
        # if cost is more than 1e6, it is out of the boundary or will collide
        elif next_cost >= 1e6:
            self.u = - 0.0 * self.u
        elif self._len_to_goal <= self.radius ** 2:
            to_goal_vector = self.goal_final - self.pos_est
            distance_to_final_goal = np.linalg.norm(to_goal_vector)
            to_goal_normal_vector = to_goal_vector / distance_to_final_goal
            vel_speed = min(self.speed_max, distance_to_final_goal / self.dt)
            self.u = to_goal_normal_vector * vel_speed
        # other is the result of the MPPi
        # else:
        #     # self.u = magnitude_angle_to_xy(action).numpy()
        #     action_numpy = action.numpy()
        #     if np.linalg.norm(action_numpy) < self.speed_max / 4:
        #         self.u = - 0.0 * self.u
        #     else:
        #         self.u = action_numpy
        else:
            # self.u = magnitude_angle_to_xy(action).numpy()
            self.u = action.numpy()
