import copy

from ratinabox import Environment
from ratinabox import utils as rutils

from predhpc import util


class NewEnv(Environment):

    default_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "scale": 1,  # scale of environment (in metres)
        "aspect": 1,  # x/y aspect ratio for the (rectangular) 2D environment
        "dx": 0.01,  # discretises the environment (for plotting purposes only)
    }

    def __init__(self, params=dict()):
    
        self.params = copy.deepcopy(__class__.default_params)     
        self.params.update(params)

        super().__init__(self.params)

    
