import numpy as np


class BaseObs:
    dim: int
    id: int

    # real state
    pos_real: np.ndarray = []

    # measured state
    pos_mea: np.ndarray = []
    pos_mea_cov: np.ndarray = []

    # estimated state
    pos_est: np.array = []
    pos_est_cov: np.ndarray = []

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_id(self, obs_id):
        self.id = obs_id

    def get_estimated_state(self):
        pass
