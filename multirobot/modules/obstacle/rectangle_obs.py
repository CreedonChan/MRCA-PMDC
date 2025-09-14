import numpy as np

from multirobot.modules.obstacle.base_obs import BaseObs


class RectangleObs(BaseObs):
    # size
    size: np.ndarray = []
    # yaw
    yaw: float

    def __init__(self, init_data: dict, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        # obs_type
        self.obs_type = 'rectangle_obs'

        # init the data
        self.set_id(init_data.get('id'))
        self.set_pos(init_data.get('pos_real'))
        self.set_pos_noise(init_data.get('pos_noise'))
        self.set_size(init_data.get('size'))
        self.set_yaw(init_data.get('yaw', 0.0))


    def set_id(self, obs_id):
        self.id = obs_id

    def set_pos(self, pos: np.ndarray):
        self.pos_real = pos

    def set_pos_noise(self, cov: np.ndarray):
        self.pos_mea_cov = cov

    def set_size(self, size: np.ndarray):
        self.size = size

    def set_yaw(self, yaw: float):
        self.yaw = yaw

    def get_measured_state(self):
        self.pos_mea = self.pos_real.copy()
        # add some noise to the measured state
        noise = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.pos_mea_cov)
        self.pos_mea += noise

    def get_estimated_state(self):
        self.get_measured_state()
        # let estimated state to be the measured one
        self.pos_est = self.pos_mea.copy()
        self.pos_est_cov = self.pos_mea_cov.copy()
