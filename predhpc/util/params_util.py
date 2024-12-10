from predhpc.util import ext_util

SCALE_LINEAR = 4.0
SCALE_TMAZE = 3.0
SCALE = 1.0
DT = 0.03

OBJ_COLOR = "#22772E"  # dark green
PC_COLOR = "#99193A"  # dark red
PYR_SOMA_COLOR = "#3D3D79"  # dark purple
PYR_DEND_COLOR = "#8787C9"  # light purple

LINEAR_SIGMOID_ACTIVATION_PARAMS = ext_util.get_standard_sigmoid_params(
    min_fr=0.0, max_fr=10.0, mid_x=6.0, width_x=8.0
)

DEND_SIGMOID_ACTIVATION_PARAMS = ext_util.get_standard_sigmoid_params(
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
    environments = ["linear", "tmaze", "openfield"]

    if environment not in environments:
        environment_strs = ", ".join([f"'{env}'" for env in environments])
        raise ValueError(
            f"Environment should have one of the following values: {environment_strs}, not '{environment}'."
        )

    return environment


def get_env_params(scale=None, environment="linear", **kwargs):
    """
    get_env_params()

    Obtain default parameters to initialise an environment (env.Environment).

    Args:
    - scale (float, optional): The scale of the environment.
    - environment (str, optional): The environment type
        (e.g., linear, tmaze, openfield).

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
            "prop_env": 0.1,
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
            "init_seed": 55,
            "scale": scale,
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
        (e.g., linear, tmaze, openfield).

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
        "speed_mean": 0.08,  # sets directionality
        "speed_std": 0.04,
    }

    if environment == "linear":
        scale = scale or SCALE_LINEAR

        agent_params["start_position"] = 0 + dt
        agent_params["reset_position"] = scale - dt
        agent_params["target_position"] = scale - dt * 8
        agent_params["fixed_direction"] = True
        agent_params["wait_between_targets"] = 30

    elif environment == "tmaze":
        agent_params["thigmotaxis"] = 0.5
        agent_params["left_arm_prop"] = 0.5
        agent_params["target_reached_within_tol_prop_to_speed_dt"] = (
            8  # very wide for target
        )

    elif environment == "openfield":
        agent_params["thigmotaxis"] = 0.5
        agent_params["num_random_walk_steps"] = 300
        agent_params["always_log_teleportation"] = True
        agent_params["no_target_factor"] = 5
        agent_params["target_reached_within_tol_prop_to_speed_dt"] = (
            8  # very wide for target
        )

    for key, value in kwargs.items():
        agent_params[key] = value

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
        "color": OBJ_COLOR,
    }

    if vector:
        Obj_params["line_of_sight"] = True

    if environment == "linear":
        Obj_params["widths"] = 0.1

    elif environment == "tmaze":
        Obj_params["widths"] = 0.1

    elif environment == "openfield":
        Obj_params["widths"] = 0.07

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
        "description": "gaussian_threshold",
        "place_cell_centres": "uniform",
        "min_fr": 0,
        "max_fr": 10,
        "color": PC_COLOR,
        "widths": 0.2,
    }

    if environment == "linear":
        n = 32 if n is None else n
        PC_params["n"] = n

    elif environment == "tmaze":
        n = 177 if n is None else n  # 40 for one row, one column
        PC_params["n"] = n
        PC_params["wall_geometry"] = "line_of_sight"  # due to environment shape

    elif environment == "openfield":
        n = 30**2 if n is None else n
        PC_params["n"] = n
        PC_params["wall_geometry"] = "line_of_sight"  # due to environment shape

    if n is not None:
        PC_params["n"] = n

    for key, value in kwargs.items():
        PC_params[key] = value

    return PC_params


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

    if two_compartment:
        Pyr_params = {
            "name": "Pyr_TwoComp",
            "n": n,
            "biases": None,
            "dend_init_weights_zero": False,
            "soma_init_weights_zero": False,
            "soma_activation_function": LINEAR_SIGMOID_ACTIVATION_PARAMS,
            "dend_activation_function": DEND_SIGMOID_ACTIVATION_PARAMS,
            "inhibit_activation_function": LINEAR_SIGMOID_ACTIVATION_PARAMS,
            "soma_apply_Ojas_rule": False,  # subtractive normalization may blow up with high clamping
            "soma_color": PYR_SOMA_COLOR,
            "dend_color": PYR_DEND_COLOR,
            "inhibit_dend": True,
            "dend_first": True,
            "soma_single_BTSP": False,
            "soma_BTSP_distance_prop": None,
            "dend_w_init_loc": 0.4,
            "soma_w_init_loc": 0.04,
            "dend_w_init_scale": 0,
            "soma_w_init_scale": 0,
            "soma_to_dend_weight": 0.2,
            "dend_to_soma_weight": 1,
            "soma_normalize_weights_divisively": True,
            "soma_regularization_alpha": 0.6,
            "soma_p": 1,
            "soma_lr": 5e-5,  # basic learning rate
            "soma_BTSP_lr_fact": 20,  # BTSP clamp
            "soma_NMDA_activation_threshold": 2,  # threshold for NMDA activation
            "soma_BTSP_induction_threshold": 8,  # sustainedrequired for BTSP
            "soma_BTSP_plateau_length": 0.12,  # plateau length required for BTSP
            "soma_BTSP_filter_tau": 4,  # BTSP kernel tau
            "soma_BTSP_trend_tau": None,  # BTSP kernel tau
            "inhibit_weight": 3.0,  # strength of dendritic inhibition from soma
            "inhibit_input_filter_tau": 3,
            "inhibit_input_trend_tau": None,
            "mutual_inhibition_weight": None,
            "lateral_tau": 0.3,
        }
    else:
        Pyr_params = {
            "name": "Pyr",
            "n": n,
            "color": PYR_SOMA_COLOR,
            "biases": None,
            "init_weights_zero": False,
            "w_init_loc": 0.1,
            "w_init_scale": 0,
            "regularization_alpha": 0.3,  # very low
            "lr": 5e-5,
            "normalize_weights_divisively": True,
            "p": 1,
        }
        if BTSP:
            Pyr_params["name"] = "Pyr_BTSP"
            Pyr_params["BTSP_lr_fact"] = 5e3  # very high clamp
            Pyr_params["BTSP_filter_tau"] = 4
            Pyr_params["BTSP_trend_tau"] = None  # BTSP kernel tau

            if NMDA:
                Pyr_params["NMDA_activation_threshold"] = (
                    2  # threshold for NMDA activation
                )
                Pyr_params["BTSP_induction_threshold"] = (
                    8  # firing rate required for BTSP
                )
                Pyr_params["BTSP_plateau_length"] = (
                    0.12  # plateau length required for BTSP
                )

    if environment == "linear":
        if two_compartment:
            pass
        else:
            if BTSP:
                if NMDA:
                    pass

    elif environment == "tmaze":
        if two_compartment:
            Pyr_params["soma_regularization_alpha"] = 0.3
            Pyr_params["inhibit_weight"] = 4.0
        else:
            Pyr_params["regularization_alpha"] = 0.3
            if BTSP:
                Pyr_params["BTSP_lr_fact"] = 8e3
                if NMDA:
                    pass

    elif environment == "openfield":
        if two_compartment:
            Pyr_params["soma_regularization_alpha"] = 0.3
            Pyr_params["inhibit_weight"] = 4.0
        else:
            Pyr_params["regularization_alpha"] = 0.3
            if BTSP:
                Pyr_params["BTSP_lr_fact"] = 5e3
                if NMDA:
                    pass

    for key, value in kwargs.items():
        Pyr_params[key] = value

    return Pyr_params
