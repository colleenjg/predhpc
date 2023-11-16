import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from ratinabox import Environment, PlaceCells
from tqdm import tqdm

from predhpc import agent, neurons, params_util


def run_linear_track(dt=None, scale=None, num_steps=1000, **CA1_kwargs):
    if dt is None:
        dt = params_util.DT

    if scale is None:
        scale = params_util.SCALE

    env_params = params_util.get_env_params(
        boundary_conditions="periodic",  # to avoid receptor binding discontinuity
    )

    agent_params = params_util.get_agent_params(
        dt=dt,
        target_position=(scale - dt * 16),
        speed_mean=0.8,
        speed_std=0.1,
    )

    EC_params = params_util.get_EC_params(dt=dt)

    CA3_PC_params = params_util.get_CA3_PC_params(dt=dt)

    Env = Environment(params=env_params)

    Ag = agent.ResetableAgent(Env, params=agent_params)

    # add objects directly
    Env.objects = {
        "objects": np.asarray(Ag.target_position).reshape(1, 1),
        "object_types": np.asarray([0]),
    }

    ECs = neurons.object_neurons.ObjectCells(Ag, params=EC_params)
    CA3_PCs = PlaceCells(Ag, params=CA3_PC_params)

    CA1_params = params_util.get_CA1_params(
        n=1,
        dt=dt,
        environment="linear",
        two_compartment=True,
        BTSP=True,
        BTSP_single=True,
        **CA1_kwargs,
    )

    CA1_params["dend_input_layers"] = [ECs]
    CA1_params["soma_input_layers"] = [CA3_PCs]

    CA1s = neurons.two_comp_neurons.TwoCompLayer(Ag, params=CA1_params)
    CA1s.set_learn(soma=False, dend=False, inhibit=False)
    CA1s.set_BTSP_learn(soma=False, dend=False)

    btsp_started = False
    for _ in tqdm(range(num_steps)):
        Ag.update()
        ECs.update()
        CA3_PCs.update()
        CA1s.update()

        if not btsp_started and len(Ag.trajectory_df) > 2:
            CA1s.set_BTSP_learn(soma=True, dend=False)
            btsp_started = True

    # compute the number of BTSP events that could have occurred
    BTSP_ramp = np.asarray(CA1s.SomaCompartment.history["BTSP_ramp"])[:, 0]
    BTSP_steps = np.where(BTSP_ramp >= 1)[0]
    num_BTSP_events = 

    # compute position where the BTSP event occurred
    if BTSP_ramp.sum() >= 1:
        BTSP

    # compute where the BTSP events would have occurred

    # compute the track


def run_hyperparameter_search(dt=None, save_directory=None, **kwargs):
    from ray import tune

    dt = params_util.DT or dt

    def objective(config):
        config["inhibit_input_filter_tau"] = dt * config.pop(
            "inhibit_input_filter_tau_fact"
        )

        kwargs.update(config)
        output_dict = run_linear_track(**kwargs)

        return output_dict

    search_space = {  # ②
        "BTSP_lr_fact": tune.grid_search([500, 600, 700, 800]),
        "inhibit_weight": tune.grid_search([1.6, 1.7, 1.8, 1.9]),
        "inhibit_input_filter_tau_fact": tune.grid_search([14, 17, 20, 23]),
    }

    tuner = tune.Tuner(objective, param_space=search_space)

    results = tuner.fit()

    df = results.get_dataframe()

    if save_directory is None:
        save_directory = Path("..", "data", "hyperparameter_search")

    # get date and time string
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    df.to_csv(Path(save_directory, f"linear_{date_time_str}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--save_directory", type=str, default=None)
    parser.add_argument("--hyperparameter_search", action="store_true")

    args = parser.parse_args()

    if args.hyperparameter_search:
        run_hyperparameter_search(**vars(args))
    else:
        run_linear_track(**vars(args))
