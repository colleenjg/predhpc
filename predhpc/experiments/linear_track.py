#!/usr/bin/env python3

import argparse
from pathlib import Path
from pprint import pprint
import time

from predhpc import run_manager
from predhpc.experiments import linear_track, metrics
from predhpc.util import params_util, hyper_util, gen_util


def get_save_directory(direc=None):
    """
    get_save_directory()

    Obtain, and create if necessary, the save directory for hyperparameter search
    results.

    Args:
    - direc (str or Path): Directory to save results in. If None, a default
        directory is used (../results/experiments). Default is None.

    Returns:
    - direc (Path): Directory to save results in.
    """

    if direc is None:
        direc = Path("..", "results", "experiments")
    direc = Path(direc)
    direc.mkdir(parents=True, exist_ok=True)

    return direc


def get_search_space(search_space="speed_PF"):
    """
    get_search_space()

    Get the search space for hyperparameter search.

    Args:
    - search_space (str, optional): Search space to use. Default is "speed_PF".

    Returns:
    - search_space (dict): Search space for hyperparameter search, with keys and values
        for each pyramidal neuron parameter to search over.
    """

    if isinstance(search_space, str):
        if search_space == "speed_PF":
            # in to out, col to row
            search_kwargs = {"speed_mean": [0.05, 0.4, 21]}
        elif search_space == "target_moved":
            search_kwargs = {
                "target_moved": [-params_util.MOVE_FAR, params_util.MOVE_FAR, 41]
            }
        else:
            raise ValueError(
                f"search_space must be 'speed_PF' or 'target_moved', but is {search_space}."
            )
    else:
        search_kwargs = search_space

    search_space = hyper_util.get_search_space(**search_kwargs)

    return search_space


def get_kwargs(experiment="speed_PF"):
    """
    get_kwargs()

    Get the keyword arguments for linear track simulations.

    Args:
    - experiment (str, optional): Experiment to run. Default is "speed_PF".

    Returns:
    - kwargs (dict): Keyword arguments for linear track simulations, with keys and
        values for each parameter.
    """

    if experiment in ["speed_PF", "target_moved"]:
        kwargs = {
            "wait_at_end": 0,
            "speed_std": 0,
            "max_num_steps": None,
            "max_num_traj": 5,
            "max_num_target_reaches": None,
            "num_repeats": 4,
            "save_name": f"linear_{experiment}",
        }
    else:
        raise ValueError(
            f"experiment must be 'speed_PF' or 'target_moved', but is {experiment}"
        )

    return kwargs


def get_Pyrs(
    scale=params_util.SCALE_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD,
    wait_at_end=params_util.WAIT_LINEAR,
    **Pyr_kwargs,
):
    """
    get_Pyrs()

    Get Pyr. layer for linear track simulation.

    Args:
    - speed_mean (float, optional): Mean speed of the agent. Default is
        params_util.SPEED_MEAN_LINEAR.
    - speed_std (float or str, optional): Standard deviation of the agent's speed.
        If a float, it is used as the standard deviation. If a string, it must be
        either "high" or "low". Default is params_util.SPEED_STD.
    - wait_at_end (float, optional): Wait time after a track run. Default is
        params_util.WAIT_LINEAR.

    Keyword args:
    - **Pyr_kwargs (dict): Keyword arguments passed to params_util.get_Pyr_params().

    Returns:W
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    """

    env_params = params_util.get_env_params(
        scale=scale,
        environment="linear",
    )

    if isinstance(speed_std, str):
        if speed_std == "high":
            div = 1
        elif speed_std == "low":
            div = 8
        else:
            raise ValueError(
                "If speed_std is not a float, it must be 'high' or 'low', but "
                f"is {speed_std}"
            )
        speed_std = speed_mean / div

    target_position = params_util.get_target_position(environment="linear", scale=scale)
    agent_params = params_util.get_agent_params(
        environment="linear",
        scale=scale,
        target_position=target_position,
        speed_mean=speed_mean,
        speed_std=speed_std,
        wait_at_end=wait_at_end,
    )

    Pyr_params = params_util.get_Pyr_params(environment="linear", **Pyr_kwargs)

    Pyrs = run_manager.init_env_objects(
        env_params=env_params,
        agent_params=agent_params,
        Pyr_params=Pyr_params,
        environment="linear",
        plot=False,
    )

    return Pyrs


def run_linear_track(
    skip_runs=1,
    max_num_traj=None,
    max_num_target_reaches=None,
    max_num_steps=5000,
    wait_at_end=params_util.WAIT_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD,
    target_moved=0,
    disable_tqdm=False,
    plot=True,
    **Pyr_kwargs,
):
    """
    run_linear_track()

    Run a linear track simulation with the agent optionally in different modes
    (consecutively).

    Args:
    - skip_runs (int, optional): Number of track runs to skip before enabling BTSP.
        Default is 1.
    - max_num_traj (int, optional): Maximum number of trajectories to run.
        Default is None.
    - max_num_target_reaches (int or None, optional): Maximum number of target
        reaches to run. Default is None.
    - max_num_steps (int or None, optional): Maximum number of steps to run. Will
        constrain other stopping conditions (number of target reaches or trajectories).
        Pass None to avoid constraining these by number of steps, and learning will
        only stop when one of those conditions are reached, if provided.
        Default is 5000.
    - wait_at_end (float, optional): Wait time after a track run. Default is
        params_util.WAIT_LINEAR.
    - speed_mean (float, optional): Mean speed of the agent. Default is
        params_util.SPEED_MEAN_LINEAR.
    - speed_std (float, optional): Standard deviation of the agent's speed.
        Default is params_util.SPEED_STD.
    - target_moved (float, optional): Amount to move the target position after the
        first set of trajectories or steps. If 0, no move is done.
        Default is 0.
    - disable_tqdm (bool, optional): Whether to disable tqdm. Default is False.
    - plot (bool, optional): Whether to generate plots. Default is True.

    Keyword args:
    - **Pyr_kwargs (dict): Keyword arguments passed to params_util.get_Pyr_params().

    Returns:
    - BTSP_metrics (dict): Dictionary of BTSP metrics, keys and values:
        - "{BTSP_metric}_({mode})": BTSP metric value for specific mode.
    if plot:
    - plot_dict (dict): Dictionary of plots, with keys and values:
        - "spatial_axes_{mode}": Spatial plots for specific mode
        - "time_axes_{mode}": Time plots for specific mode
    """

    Pyrs = get_Pyrs(
        wait_at_end=wait_at_end,
        speed_mean=speed_mean,
        speed_std=speed_std,
        **Pyr_kwargs,
    )

    learning_runs = ["initial"]
    if target_moved != 0:
        learning_runs.append("moved")

    run_kwargs = {
        "max_num_traj": max_num_traj,
        "max_num_steps": max_num_steps,
        "max_num_target_reaches": max_num_target_reaches,
    }

    learner = Pyrs
    for i, learning_run in enumerate(learning_runs):
        prev = Pyrs.Agent.get_num_completed_trajectories()

        use_run_kwargs = run_kwargs.copy()
        for key in list(run_kwargs.keys()):
            if run_kwargs[key] is not None:
                use_run_kwargs[key] = run_kwargs[key] * (i + 1)

        if learning_run == "moved":
            Pyrs.Agent.move_target_position(target_moved)

        use_plot = plot and (i == len(learning_runs) - 1)
        outputs = run_manager.learn_1D_BTSP(
            Pyrs_or_learner=learner,
            use_Hebbian=False,
            BTSP_on=skip_runs + 1,
            record_weights_at_BTSP=False,
            no_logs=disable_tqdm,
            plot=use_plot,
            **use_run_kwargs,
        )
        learner = outputs[0] if plot else outputs

        num_traj_completed = Pyrs.Agent.get_num_completed_trajectories() - prev
        if num_traj_completed < skip_runs + 2:
            raise RuntimeError(f"Only {num_traj_completed} trajectories completed.")

        if learning_run == "moved":
            Pyrs.Agent.move_target_position(-target_moved)  # return

    if plot:
        plot_dict = {
            "spatial_axes": outputs[1],
            "time_axes": outputs[2],
        }

    BTSP_metrics = metrics.compute_BTSP_metrics(Pyrs)

    if plot:
        return BTSP_metrics, plot_dict
    else:
        return BTSP_metrics


def run_linear_experiment_grid(
    search_space,
    max_num_traj=None,
    max_num_target_reaches=None,
    max_num_steps=5000,
    direc=None,
    num_CPUs=4,
    num_repeats=4,
    save_name="linear",
    disable_tqdm=True,
    plot=False,
    debug=False,
    **Pyr_kwargs,
):
    """
    run_linear_experiment_grid()

    Runs a grid of linear track simulations with different hyperparameters.

    Args:
    - search_space (dict or str): Search space for hyperparameter search, with keys
        and values for each pyramidal neuron parameter to search over.
    - max_num_traj (int, optional): Maximum number of trajectories to run.
        Default is None.
    - max_num_target_reaches (int or None, optional): Maximum number of target
        reaches to run. Default is None.
    - max_num_steps (int or None, optional): Maximum number of steps to run. Will
        constrain other stopping conditions (number of target reaches or trajectories).
        Pass None to avoid constraining these by number of steps, and learning will
        only stop when one of those conditions are reached, if provided.
        Default is 5000.
    - direc (str, optional): Directory to save results in. If None, a default
        directory is used (see hyper_util.get_save_directory()). Default is None.
    - num_CPUs (int, optional): Number of CPUs to run search across. Default is 4.
    - num_repeats (int, optional): Number of repeats for each hyperparameter set.
        Default is 4.
    - save_name (str, optional): Name to save the results under. Default is
        "linear".
    - disable_tqdm (bool, optional): Whether to disable tqdm. Default is True.
    - plot (bool, optional): Whether to generate plots. Default is False.
    - debug (bool, optional): Whether to run in debug mode. Default is False.

    Keyword args:
    - **kwargs (dict): Keyword arguments passed to get_Pyrs().
    """

    def objective(config):
        """
        objective(config)

        Objective function for a hyperparameter search run.

        Args:
        - config (dict): Configuration dictionary, specifying parameters for a specific
            run.

        Returns:
        - output_dict (dict): Output dictionary, with metrics for the run.
        """

        kwargs_use = Pyr_kwargs.copy()
        kwargs_use.update(config)

        output_dict = run_linear_track(
            max_num_traj=max_num_traj,
            max_num_target_reaches=max_num_target_reaches,
            max_num_steps=max_num_steps,
            disable_tqdm=disable_tqdm,
            plot=plot,
            **kwargs_use,
        )

        return output_dict

    hyper_util.run_hyperparameter_search(
        objective,
        search_space,
        direc=direc,
        save_name=save_name,
        num_CPUs=num_CPUs,
        num_repeats=num_repeats,
        debug=debug,
    )


def get_args():
    """
    get_args()

    Get command line arguments for speed vs PF width simulations.

    Returns:
    - args (argparse.Namespace): Parsed command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", default="speed_PF")
    parser.add_argument("--direc", type=Path, default=None)
    parser.add_argument("--num_CPUs", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--replotting_path", type=Path, default=None)

    args = parser.parse_args()

    return args


def main():
    """
    main()

    Main function for linear track simulations.
    """

    args = get_args()

    if args.replotting_path is not None:
        start_time = time.perf_counter()
        hyper_util.replot_from_csvs(args.replotting_path)
        gen_util.get_duration_str(start_time, log=True)

    else:
        start_time = time.perf_counter()

        search_space = get_search_space(search_space=args.experiment)
        kwargs = get_kwargs(experiment=args.experiment)
        direc = get_save_directory(direc=args.direc)

        linear_track.run_linear_experiment_grid(
            search_space=search_space,
            direc=direc,
            num_CPUs=args.num_CPUs,
            debug=args.debug,
            plot=False,
            **kwargs,
        )

        gen_util.get_duration_str(start_time, log=True)


if __name__ == "__main__":
    main()
