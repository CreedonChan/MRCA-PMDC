from copy import copy

import torch
import numpy as np
from mpmath.math2 import sqrt2
from numpy.f2py.auxfuncs import throw_error

from multirobot.modules.robot.base_robot import BaseRobot
from torch.distributions import Normal


class PassiveRobot(BaseRobot):
    # control max
    speed_max = 0.0

    def __init__(self, init_data: dict, **kwargs):
        super().__init__(**kwargs)

        # robot type
        self.robot_type = 'passive'

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

    def simulate_one_step(self):
        self.pos_real = self.pos_real + self.u * self.dt
        self.speed_real = self.u

        # this is for the travelled distance
        self.travelled_distance = self.travelled_distance + np.linalg.norm(self.u * self.dt)

    def calculate_control_input(self):
        to_goal_vector = self.goal_final - self.pos_real
        distance_to_final_goal = np.linalg.norm(to_goal_vector)

        # if arrived, don't move
        if self.is_arrived:
            self.u = - 0.0 * self.u
        else:
            to_goal_normal_vector = to_goal_vector / distance_to_final_goal
            vel_speed = min(self.speed_max, distance_to_final_goal / self.dt)
            self.u = to_goal_normal_vector * vel_speed