import numpy as np

from multirobot.modules.obstacle.base_obs import BaseObs


class CircleObs(BaseObs):

    # radius
    radius: float

    def __init__(self, init_data: dict, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        # obs_type
        self.obs_type = 'circle_obs'

        # init the data
        self.set_id(init_data.get('id'))
        self.set_pos(init_data.get('pos_real'))
        self.set_radius(init_data.get('radius'))

    def set_id(self, obs_id):
        self.id = obs_id

    def set_pos(self, pos: np.ndarray):
        self.pos_real = pos

    def set_radius(self, radius):
        self.radius = radius

    def get_estimated_state(self):
        # let estimated state to be the real one, we just use a circle to describe the obstacle
        self.pos_est = self.pos_real

    def collision_checking(self, robot_pos_real: np.array, robot_radius):
        distance_vector = self.pos_real - np.array(robot_pos_real)
        threshold = self.radius + robot_radius
        if np.linalg.norm(distance_vector) < threshold:
            return True
        return False
