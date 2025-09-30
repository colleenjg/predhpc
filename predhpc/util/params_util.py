import copy

import numpy as np
from ratinabox import utils as rutils

from predhpc.util import gen_util, ext_util

# Scale parameters
SCALE_LINEAR = 6.0
SCALE_TMAZE = 4.0
SCALE = 2.0

# Target parameters
REL_TARGET_POS = 3 / 5
SHIFT_CLOSE = 0.2
SHIFT_MID = 0.7
SHIFT_FAR = 2.0

# Spatial proximity parameters
TOLERANCE_LINEAR = 0.55
TOLERANCE_2D = 5

# Temporal parameters
DT = 0.03
WAIT_LINEAR_SEC = 20
WAIT_LINEAR = int(WAIT_LINEAR_SEC / DT)

# Speed parameters
SPEED_MEAN_LINEAR = 0.25  # m/s
SPEED_STD = 0.05  # m/s
SPEED_MEAN_2D = 0.28  # m/s (mean tends to be undershot in practice)

# Place cell parameter
PC_SIGMA = 0.1  # x2.35 for width at 50% of peak (FWHM), x4.29 for 10% of peak

# BTSP filter parameters
PRE_BTSP_FILTER_TAU_POS = 2.0
PRE_BTSP_NEG_FILTER_DELTA = 0.010
PRE_BTSP_NEG_FILTER_WEIGHT = 1.0010

POST_BTSP_FILTER_TAU_POS = 1.4
POST_BTSP_NEG_FILTER_DELTA = 0.010
POST_BTSP_NEG_FILTER_WEIGHT = 1.0006

PRE_BTSP_FILTER_TAU_NEG = PRE_BTSP_FILTER_TAU_POS + PRE_BTSP_NEG_FILTER_DELTA
POST_BTSP_FILTER_TAU_NEG = POST_BTSP_FILTER_TAU_POS + POST_BTSP_NEG_FILTER_DELTA

# Color parameters
TARGET_COLOR = "#3C539B"  # blue
NOVEL_COLOR = "#22772E"  # dark green

OBJ_COLOR = TARGET_COLOR
PC_COLOR = "#99193A"  # dark red
PYR_SOMATIC_COLOR = "#8787C9"  # light purple
PYR_APICAL_COLOR = "#3D3D79"  # dark purple

# Activation function parameters
LINEAR_SIGMOID_ACTIVATION_PARAMS = ext_util.get_standard_sigmoid_params(
    min_fr=0.0, max_fr=10.0, mid_x=6.0, width_x=8.0
)

APICAL_SIGMOID_ACTIVATION_PARAMS = ext_util.get_standard_sigmoid_params(
    min_fr=0.0, max_fr=10.0, mid_x=5.0, width_x=6.0
)


def check_environment(environment="linear"):
    """
    check_environment()

    Check if the environment type is valid.

    Parameters:
    - environment (str, optional): The environment to check.

    Raises:
    - ValueError: If the environment is not one of the valid environments.

    Returns:
    - str: The environment type, set to lowercase.
    """

    environment = environment.lower()
    environments = ["linear", "tmaze", "openfield", "openfield_corridor"]

    if environment not in environments:
        environment_strs = ", ".join([f"'{env}'" for env in environments])
        raise ValueError(
            f"Environment should have one of the following values: {environment_strs}, not '{environment}'."
        )

    return environment


def get_target_position(environment="linear", scale=SCALE_LINEAR):
    """
    get_target_position()

    Get target position.

    Returns:
    - target_position (1D np.ndarray): Target position.
    """

    if environment == "linear":
        target_position = np.asarray([scale * REL_TARGET_POS])
    else:
        raise NotImplementedError(
            "Function only implemented for the linear environment."
        )
    return target_position


def get_activation_function(activation_function=None):
    """
    get_activation_function()

    Obtain the activation function to use for a neuron layer.

    Args:
    - activation_function (str or dict, optional): The activation function to use.
        If a string, the activation function is assumed to be a standard activation
        function (e.g., "sigmoid", "relu"). If a dictionary, the dictionary should
        contain the activation function and any additional parameters
        (i.e., the keys: "activation", "min_fr", "max_fr", "mid_x", "width_x").

    Returns:
    - activation_function (function): The activation function to use.
    """

    if activation_function is None:
        activation_function = LINEAR_SIGMOID_ACTIVATION_PARAMS

    elif isinstance(activation_function, str):
        activation_function = {"activation": activation_function}

    if isinstance(activation_function, dict):
        activation_params = copy.deepcopy(activation_function)
        activation_function = lambda x, deriv=False: rutils.activate(
            x, deriv=deriv, other_args=activation_params
        )

    return activation_function


def get_default_BTSP_filter_param_dict(incl_BTSP_str=True, neg_delta=False):
    """
    get_default_BTSP_filter_param_dict()

    Obtain default parameters for BTSP filters.

    Args:
    - incl_BTSP_str (bool, optional): If True, "BTSP_" is included in the keys.
        If False, it is omitted. Defaults to True.

    Returns:
    - BTSP_filter_param_dict (dict): Default parameters for BTSP filters.
    """

    BTSP_str = "_BTSP" if incl_BTSP_str else ""

    BTSP_filter_param_dict = {
        f"pre{BTSP_str}_filter_tau_pos": PRE_BTSP_FILTER_TAU_POS,
        f"pre{BTSP_str}_neg_weight": PRE_BTSP_NEG_FILTER_WEIGHT,
        f"post{BTSP_str}_filter_tau_pos": POST_BTSP_FILTER_TAU_POS,
        f"post{BTSP_str}_neg_weight": POST_BTSP_NEG_FILTER_WEIGHT,
    }

    if neg_delta:
        BTSP_filter_param_dict[f"pre{BTSP_str}_neg_delta"] = PRE_BTSP_NEG_FILTER_DELTA
        BTSP_filter_param_dict[f"post{BTSP_str}_neg_delta"] = POST_BTSP_NEG_FILTER_DELTA
    else:
        BTSP_filter_param_dict[f"pre{BTSP_str}_filter_tau_neg"] = (
            PRE_BTSP_FILTER_TAU_NEG
        )
        BTSP_filter_param_dict[f"post{BTSP_str}_filter_tau_neg"] = (
            POST_BTSP_FILTER_TAU_NEG
        )

    return BTSP_filter_param_dict


def get_env_params(scale=None, environment="linear", **kwargs):
    """
    get_env_params()

    Obtain default parameters to initialise an environment (env.Environment).

    Args:
    - scale (float, optional): The scale of the environment.
    - environment (str, optional): The environment type
        (e.g., linear, tmaze, openfield, openfield_corridor).

    Keyword args:
    - **kwargs: Additional environment parameters to include (can overwrite default
        parameters).

    Returns:
    - env_params (dict): Environment initialisation parameters.
    """

    environment = check_environment(environment)

    if environment == "linear":
        scale = scale or SCALE_LINEAR
        env_params = {
            "dimensionality": "1D",
            "scale": scale,
            "boundary_conditions": "periodic",
        }

    elif environment == "tmaze":
        scale = scale or SCALE_TMAZE
        env_params = {
            "prop_env": 0.3 / scale,
            "scale_x": scale,
            "scale_y": scale,
        }

    elif environment == "openfield":
        scale = scale or SCALE
        env_params = {
            "init_random_walls": 5,
            "init_random_reward_obj": 4,
            "init_random_novel_obj": 4,
            "init_random_teleport_pairs": 6,
            "wall_lengths": [0.1 * SCALE, 0.2 * SCALE],
            "init_seed": 75,
            "scale": scale,
            "dx": scale / 100,
        }

    elif environment == "openfield_corridor":
        scale = scale or SCALE

        locations = {
            "reward": [0.25, 0.1],
            "teleport_in_out": [[0.6, 0.5], [0.2, 0.1]],
            "wall_coords": [[0, 0.2], [0.6, 0.2]],
        }
        locations = {key: np.asarray(data) * scale for key, data in locations.items()}

        env_params = {
            "init_random_reward_obj": 0,
            "init_random_novel_obj": 0,
            "init_random_walls": 0,
            "init_random_teleport_pairs": 0,
            "walls": [locations["wall_coords"]],
            "init_reward_obj": [locations["reward"]],
            "init_teleport_pairs": [locations["teleport_in_out"]],
            "scale": scale,
            "dx": scale / 100,
        }

    for key, value in kwargs.items():
        env_params[key] = value

    return env_params


def get_agent_params(dt=DT, scale=None, environment="linear", **kwargs):
    """
    get_agent_params()

    Obtain default parameters to initialise an agent (agent.ResetableAgent).

    Args:
    - dt (float, optional): The time step.
    - scale (float, optional): The scale of the environment, used for linear agent.
    - environment (str, optional): The environment type
        (e.g., linear, tmaze, openfield, openfield_corridor).

    Keyword args:
    - **kwargs: Additional agent parameters to include (can overwrite default
        parameters).

    Returns:
    - agent_params (dict): Agent initialisation parameters.
    """

    environment = check_environment(environment)

    dt = dt or DT

    agent_params = {
        "dt": dt,
        "head_direction_smoothing_timescale": dt * 2,
        "wait_between_same_target": 30,
    }

    if environment == "linear":
        scale = scale or SCALE_LINEAR

        agent_params["speed_mean"] = SPEED_MEAN_LINEAR
        agent_params["speed_std"] = SPEED_STD
        agent_params["start_position"] = 0 + dt
        agent_params["reset_position"] = scale - dt
        agent_params["target_position"] = scale - dt * 8
        agent_params["fixed_direction"] = True
        agent_params["wait_after_trajectory"] = int(WAIT_LINEAR_SEC / dt)
        agent_params["target_reached_within_tol_prop_to_speed_dt"] = TOLERANCE_LINEAR
        agent_params["reset_reached_within_tol_prop_to_speed_dt"] = TOLERANCE_LINEAR

    elif environment == "tmaze":
        agent_params["speed_mean"] = SPEED_MEAN_2D
        agent_params["thigmotaxis"] = 0.5
        agent_params["left_arm_prop"] = 0.5
        agent_params["target_reached_within_tol_prop_to_speed_dt"] = TOLERANCE_2D
        agent_params["reset_reached_within_tol_prop_to_speed_dt"] = TOLERANCE_2D

    elif environment in ["openfield", "openfield_corridor"]:
        agent_params["speed_mean"] = SPEED_MEAN_2D
        agent_params["thigmotaxis"] = 0.5
        agent_params["num_random_walk_steps"] = 300
        agent_params["always_log_teleportation"] = True
        agent_params["no_target_factor"] = 5
        agent_params["target_reached_within_tol_prop_to_speed_dt"] = TOLERANCE_2D
        if environment == "openfield_corridor":
            agent_params["reward_factor"] = 3
            agent_params["no_target_factor"] = 1

    for key, value in kwargs.items():
        agent_params[key] = value

    # 2D environments use speed_mean as the Rayleigh sigma, unless speed_std is 0
    if (
        environment in ["tmaze", "openfield", "openfield_corridor"]
        and "speed_mean" in agent_params.keys()
    ):
        if "speed_std" in agent_params.keys() and agent_params["speed_std"] == 0:
            pass
        else:
            agent_params["speed_mean"] = gen_util.get_rayleigh_sigma(
                agent_params["speed_mean"]
            )

    return agent_params


def get_Obj_params(n=None, environment="linear", vector=False, **kwargs):
    """
    get_Obj_params()

    Obtain default parameters to initialise an object layer ("Obj")
    (object_neurons.ObjectCells).

    Args:
    - n (int, optional): The number of object neurons.
    - environment (str, optional): The environment type
        (e.g., linear, tmaze, openfield).
    - vector (bool, optional): Whether to use a vector representation.

    Keyword args:
    - **kwargs: Additional Obj. layer parameters to include (can overwrite default
        parameters).

    Returns:
    - Obj_params (dict): Obj. layer initialisation parameters.
    """

    environment = check_environment(environment)

    Obj_params = {
        "name": "Obj",
        "description": "gaussian",
        "min_fr": 0,
        "max_fr": 10,
        "widths": PC_SIGMA / 2,  # for Gaussian fields, provide sigma (st. dev.)
        "color": OBJ_COLOR,
    }

    if environment == "openfield_corridor":
        Obj_params["num_novel"] = 0
        Obj_params["num_teleport"] = 0
        Obj_params["num_reward"] = 1

    if vector:
        Obj_params["line_of_sight"] = True

    if n is not None:
        Obj_params["n"] = n

    for key, value in kwargs.items():
        Obj_params[key] = value

    return Obj_params


def get_PC_params(n=None, environment="linear", **kwargs):
    """
    get_PC_params()

    Obtain default parameters to initialise place cells (riab_neurons.PlaceCells).

    Args:
    - n (int, optional): The number of place cells.
    - environment (str, optional): The environment type
        (e.g., linear, tmaze, openfield).

    Keyword args:
    - **kwargs: Additional place cell parameters to include (can overwrite default
        parameters).

    Returns:
    - PC_params (dict): Place cell initialisation parameters.
    """

    environment = check_environment(environment)

    PC_params = {
        "name": "PCs",
        "description": "gaussian",
        "place_cell_centers": "uniform",
        "min_fr": 0,
        "max_fr": 10,
        "color": PC_COLOR,
        "widths": PC_SIGMA,  # for Gaussian fields, provide sigma (st. dev.)
    }

    if environment == "linear":
        n = n or SCALE_LINEAR * 10
        PC_params["n"] = int(n)

    elif environment == "tmaze":
        n = n or (SCALE_TMAZE * 10 * 2 * 3 - 9)
        PC_params["n"] = int(n)
        PC_params["wall_geometry"] = "line_of_sight"  # due to environment shape

    elif environment in ["openfield", "openfield_corridor"]:
        n = n or (SCALE * 10) ** 2
        PC_params["n"] = int(n)
        PC_params["wall_geometry"] = "line_of_sight"  # due to environment shape

    for key, value in kwargs.items():
        PC_params[key] = value

    return PC_params


def get_VN_params(peak, **kwargs):
    """
    get_VN_params()

    Obtain default parameters to initialise simple value neuron layer.

    Args:
    - peak (float or 1D np.ndarray): Peak value location [x, y].

    Keyword args:
    - **kwargs: Additional VN parameters to include (can overwrite default
        parameters).

    Returns:
    - VN_params (dict): VN initialisation parameters.
    """

    VN_params = {
        "name": "VN",
        "n": 1,
        "peak": np.asarray(peak),
        "description": "gaussian",
        "min_fr": 0,
        "max_fr": 10,
        "widths": 0.6,
    }

    for key, value in kwargs.items():
        VN_params[key] = value

    return VN_params


def get_Pyr_params(
    n=1, environment="linear", two_compartment=True, BTSP=True, NMDA=True, **kwargs
):
    """
    get_Pyr_params()

    Obtain default parameters to initialise Pyr. neurons (Pyr.)
    (learning_neurons.HebbianLayer).

    Args:
    - n (int, optional): The number of Pyr. neurons.
    - environment (str, optional): The environment type
        (e.g., linear, tmaze, openfield).
    - two_compartment (bool, optional): Whether to use a two-compartment model.
    - BTSP (bool, optional): If not two_compartment, whether to include
        learning_neurons.BTSPLayer parameters.
    - NMDA (bool, optional): If not two_compartment, but BTSP is True,
        whether to include learning_neurons.NMDALayer parameters.

    Keyword args:
    - **kwargs: Additional Pyr. parameters to include (can overwrite default
        parameters).

    Raises:
    - ValueError: If NMDA is True, but BTSP is False.
    - ValueError: If two_compartment is True, but BTSP or NMDA is False.

    Returns:
    - Pyr_params (dict): Pyr. initialisation parameters.
    """

    environment = check_environment(environment)

    if NMDA and not BTSP:
        raise ValueError("NMDA can only be used with BTSP.")
    if two_compartment and not (BTSP and NMDA):
        raise ValueError("Two-compartment model requires BTSP and NMDA.")

    BIASES = None
    INIT_WEIGHTS_ZERO = False
    W_INIT_SCALE = 0
    NORM_WEIGHTS_DIV = True
    LR = 2e-5
    P = 4

    if environment == "linear":
        W_INIT_LOC = 0.1
        REG_ALPHA = 2.5
        BTSP_LR = 0.2
    else:
        W_INIT_LOC = 0.04
        REG_ALPHA = 4.25
        BTSP_LR = 0.15

    if two_compartment:
        Pyr_params = {
            "name": "Pyr_TwoComp",
            "n": n,
            "biases": BIASES,
            "apical_init_weights_zero": False,
            "somatic_init_weights_zero": INIT_WEIGHTS_ZERO,
            "somatic_activation_function": LINEAR_SIGMOID_ACTIVATION_PARAMS,
            "apical_activation_function": APICAL_SIGMOID_ACTIVATION_PARAMS,
            "inhibitory_activation_function": LINEAR_SIGMOID_ACTIVATION_PARAMS,
            "somatic_apply_Ojas_rule": False,  # subtractive normalization may blow up with high clamping
            "somatic_color": PYR_SOMATIC_COLOR,
            "apical_color": PYR_APICAL_COLOR,
            "inhibitory_apical": True,
            "apical_first": True,
            "somatic_single_BTSP": False,
            "somatic_BTSP_distance_prop": None,
            "apical_w_init_loc": 0.4,
            "somatic_w_init_loc": W_INIT_LOC,
            "apical_w_init_scale": 0,
            "somatic_w_init_scale": W_INIT_SCALE,
            "somatic_to_apical_weight": 0.2,
            "apical_to_somatic_weight": 1.0,
            "somatic_normalize_weights_divisively": NORM_WEIGHTS_DIV,
            "somatic_regularization_alpha": REG_ALPHA,
            "somatic_p": P,
            "somatic_lr": LR,  # basic learning rate
            "somatic_BTSP_lr": BTSP_LR,  # BTSP learning rate
            "somatic_NMDA_activation_threshold": 2,  # threshold for NMDA activation
            "somatic_BTSP_induction_threshold": 8,  # sustained required for BTSP
            "somatic_BTSP_plateau_length": 0.12,  # plateau length required for BTSP
            "inhibitory_weight": 1.0,  # strength of apical inhibition from somatic compartment
            "inhibitory_input_filter_tau": 0.3,
            "inhibitory_input_trend_tau": None,
            "lateral_inhibition_weight": None,
            "lateral_tau": 0.3,
        }

        BTSP_filter_param_dict = get_default_BTSP_filter_param_dict()
        for key, value in BTSP_filter_param_dict.items():
            Pyr_params[f"somatic_{key}"] = value

    else:
        Pyr_params = {
            "name": "Pyr",
            "n": n,
            "color": PYR_SOMATIC_COLOR,
            "biases": BIASES,
            "init_weights_zero": INIT_WEIGHTS_ZERO,
            "w_init_loc": W_INIT_LOC,
            "w_init_scale": W_INIT_SCALE,
            "p": P,
            "normalize_weights_divisively": NORM_WEIGHTS_DIV,
            "regularization_alpha": REG_ALPHA,
        }
        if NMDA:
            Pyr_params["lr"] = LR
        else:
            Pyr_params["lr"] = LR * 1.5

        if BTSP:
            Pyr_params["name"] = "Pyr_BTSP"

            BTSP_filter_param_dict = get_default_BTSP_filter_param_dict()
            for key, value in BTSP_filter_param_dict.items():
                Pyr_params[key] = value

            if NMDA:
                Pyr_params["BTSP_lr"] = BTSP_LR
                Pyr_params["NMDA_activation_threshold"] = (
                    2  # threshold for NMDA activation
                )
                Pyr_params["BTSP_induction_threshold"] = (
                    8  # firing rate required for BTSP
                )
                Pyr_params["BTSP_plateau_length"] = (
                    0.12  # plateau length required for BTSP
                )
            else:
                Pyr_params["BTSP_lr"] = BTSP_LR * 1.5

    for key, value in kwargs.items():
        Pyr_params[key] = value

    return Pyr_params
