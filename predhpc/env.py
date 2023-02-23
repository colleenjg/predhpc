import copy

from ratinabox import Environment
from ratinabox import utils as rat_utils

DEFAULT_PARAMS = {
    "dimensionality": "2D",  # 1D or 2D environment
    "boundary_conditions": "solid",  # solid vs periodic
    "scale": 1,  # scale of environment (in metres)
    "aspect": 1,  # x/y aspect ratio for the (rectangular) 2D environment
    "dx": 0.01,  # discretises the environment (for plotting purposes only)
}

class NewEnv(Environment):
    def __init__(self, params=dict()):
    
        use_params = copy.deepcopy(DEFAULT_PARAMS)
        use_params.update(params)

        super().__init__(use_params)

    
