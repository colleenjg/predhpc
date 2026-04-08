#!/usr/bin/env python3

import argparse
from pathlib import Path
import time

from predhpc import run_manager
from predhpc.experiments import metrics
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
            search_kwargs = {"speed_mean": [0.05, 0.4, 29]}
        elif search_space == "object_shift":
            search_kwargs = {"object_shift": [-3.6, 2.4, 61]}
        else:
            raise ValueError(
                f"search_space must be 'speed_PF' or 'object_shift', but is {search_space}."
            )
    else:
        search_kwargs = search_space

    search_space = hyper_util.get_search_space(**search_kwargs)

    return search_space


def get_kwargs(experiment="speed_PF", speed_std=0):
    """
    get_kwargs()

    Get the keyword arguments for linear track simulations.

    Args:
    - experiment (str, optional): Experiment to run. Default is "speed_PF".

    Returns:
    - kwargs (dict): Keyword arguments for linear track simulations, with keys and
        values for each parameter.
    """

    if experiment in ["speed_PF", "object_shift"]:
        kwargs = {
            "wait_after_trajectory": 0,
            "speed_std": speed_std,
            "num_steps_can_stop": None,
            "num_traj_can_stop": 10,
            "num_target_reaches_can_stop": None,
            "num_repeats": 4,
            "save_name": f"linear_{experiment}",
        }

        if speed_std != 0:
            kwargs["save_name"] = f"{kwargs['save_name']}_std_{speed_std}"
    else:
        raise ValueError(
            f"experiment must be 'speed_PF' or 'object_shift', but is {experiment}"
        )

    return kwargs


def get_param_str(experiment="speed_PF", speed_std=0, log=False):
    """
    get_param_str()

    Get a parameter string based on the parameters used.

    Args:
    - experiment (str, optional): The experiment to run. Default is "speed_PF".
    - speed_std (float, optional): The speed standard deviation to use. Default is 0.
    - log (bool, optional): Whether to print the parameter string. Default is False.

    Returns:
    - param_str (str): Parameter string based on the parameters used.
    """

    if experiment == "speed_PF":
        param_str = "speed vs PF width experiment"
    elif experiment == "object_shift":
        param_str = "object shift experiment"
    else:
        raise ValueError(
            f"experiment must be 'speed_PF' or 'object_shift', but is {experiment}"
        )

    param_str = f"{param_str} (speed std: {speed_std} m/s)"

    if log:
        print(f"Running {param_str}.")

    return param_str


def get_Pyrs(
    scale=params_util.SCALE_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD_LINEAR,
    wait_after_trajectory=params_util.WAIT_LINEAR,
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
        either "high" or "low". Default is params_util.SPEED_STD_LINEAR.
    - wait_after_trajectory (float, optional): Number of steps to wait after completing
        a trajectory. Default is params_util.WAIT_LINEAR.

    Keyword args:
    - **Pyr_kwargs (dict): Keyword arguments passed to params_util.get_Pyr_params().

    Returns:W
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    """

    env_params = params_util.get_env_params(
        scale=scale,
        environment="linear",
        init_env_object_prop=params_util.REL_ENV_OBJECT_POS,
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

    agent_params = params_util.get_agent_params(
        environment="linear",
        scale=scale,
        speed_mean=speed_mean,
        speed_std=speed_std,
        wait_after_trajectory=wait_after_trajectory,
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
    num_traj_can_stop=8,
    num_target_reaches_can_stop=None,
    num_steps_can_stop=None,
    wait_after_trajectory=params_util.WAIT_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD_LINEAR,
    object_shift=0,
    disable_tqdm=False,
    plot=True,
    PF_kwargs=dict(),
    **Pyr_kwargs,
):
    """
    run_linear_track()

    Run a linear track simulation with the agent optionally in different modes
    (consecutively).

    Args:
    - skip_runs (int, optional): Number of trajectories to skip before enabling BTSP.
        Default is 1.
    - num_traj_can_stop (int, optional): Number of trajectories to run after which
        early stopping may be triggered. Default is 8.
    - num_target_reaches_can_stop (int or None, optional): Number of target
        reaches after which early stopping may be triggered. Default is None.
    - num_steps_can_stop (int or None, optional): Number of steps after which early
        stopping can occur. May prevent the learner object from reaching its other
        stopping conditions (number of target reaches or trajectories). Pass None to
        avoid constraining these by number of steps, and early stopping will only be
        triggered when one (either) of those conditions is reached, if provided.
        Default is None.
    - wait_after_trajectory (float, optional): Number of steps to wait after completing
        a trajectory. Default is params_util.WAIT_LINEAR.
    - speed_mean (float, optional): Mean speed of the agent. Default is
        params_util.SPEED_MEAN_LINEAR.
    - speed_std (float, optional): Standard deviation of the agent's speed.
        Default is params_util.SPEED_STD_LINEAR.
    - object_shift (float, optional): Amount to shift the object position after
        the first set of trajectories or steps. If 0, no shift is done.
        Default is 0.
    - disable_tqdm (bool, optional): Whether to disable tqdm. Default is False.
    - plot (bool, optional): Whether to generate plots. Default is True.
    - PF_kwargs (dict, optional): Keyword arguments passed to place field analysis
        functions. Default is an empty dictionary.

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
        wait_after_trajectory=wait_after_trajectory,
        speed_mean=speed_mean,
        speed_std=speed_std,
        **Pyr_kwargs,
    )

    learning_runs = ["initial"]
    if object_shift != 0:
        learning_runs.append("shifted")

    run_kwargs = {
        "num_traj_can_stop": num_traj_can_stop,
        "num_steps_can_stop": num_steps_can_stop,
        "num_target_reaches_can_stop": num_target_reaches_can_stop,
    }

    learner = Pyrs
    for i, learning_run in enumerate(learning_runs):
        prev = Pyrs.Agent.get_num_completed_trajectories()

        use_run_kwargs = run_kwargs.copy()
        for key in list(run_kwargs.keys()):
            if run_kwargs[key] is not None:
                use_run_kwargs[key] = run_kwargs[key] * (i + 1)

        if learning_run == "shifted":
            Pyrs.Agent.shift_target_position(object_shift)

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

        if learning_run == "shifted":
            Pyrs.Agent.shift_target_position(-object_shift)  # return

    if plot:
        plot_dict = {
            "spatial_axes": outputs[1],
            "time_axes": outputs[2],
        }

    BTSP_metrics = metrics.compute_BTSP_metrics(Pyrs, **PF_kwargs)

    if plot:
        return BTSP_metrics, plot_dict
    else:
        return BTSP_metrics


def run_linear_experiment_grid(
    search_space,
    num_traj_can_stop=10,
    num_target_reaches_can_stop=None,
    num_steps_can_stop=None,
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
    - num_traj_can_stop (int, optional): Number of trajectories to run after which
        early stopping may occur. Default is None.
    - num_target_reaches_can_stop (int or None, optional): Number of target
        reaches after which early stopping may be triggered. Default is 8.
    - num_steps_can_stop (int or None, optional): Number of steps after which early
        stopping can occur. May prevent the learner object from reaching its other
        stopping conditions (number of target reaches or trajectories). Pass None to
        avoid constraining these by number of steps, and early stopping will only be
        triggered when one (either) of those conditions is reached, if provided.
        Default is None.
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
    - **Pyr_kwargs (dict): Keyword arguments passed to get_Pyrs().
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
            num_traj_can_stop=num_traj_can_stop,
            num_target_reaches_can_stop=num_target_reaches_can_stop,
            num_steps_can_stop=num_steps_can_stop,
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


def yield_cycle_kwargs(experiment="speed_PF", speed_std=0, cycle_all=False):
    """
    yield_cycle_kwargs()

    Yield keyword arguments for linear track simulations.

    Args:
    - experiment (str, optional): The experiment to run. Default is "speed_PF".
    - speed_std (float, optional): The speed standard deviation to use. Default is 0.
    - cycle_all (bool, optional): Whether to cycle through all possible combinations of
        keyword arguments. Default is False.

    Yields:
    - kwargs (dict): Keyword arguments for linear track simulation.
    """

    if cycle_all:
        for speed_std in [0, 0.05]:
            for experiment in ["speed_PF", "object_shift"]:
                kwargs = {
                    "experiment": experiment,
                    "speed_std": speed_std,
                }
                yield kwargs

    else:
        kwargs = {
            "experiment": experiment,
            "speed_std": speed_std,
        }
        yield kwargs


def get_args():
    """
    get_args()

    Get command line arguments for speed vs PF width simulations.

    Returns:
    - args (argparse.Namespace): Parsed command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", default="speed_PF")
    parser.add_argument("--speed_std", default=0, type=float)
    parser.add_argument("--cycle_all", action="store_true")
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
        all_run_kwargs = yield_cycle_kwargs(
            experiment=args.experiment,
            speed_std=args.speed_std,
            cycle_all=args.cycle_all,
        )

        for run_kwargs in all_run_kwargs:
            start_time = time.perf_counter()

            search_space = get_search_space(search_space=run_kwargs["experiment"])
            kwargs = get_kwargs(**run_kwargs)
            direc = get_save_directory(direc=args.direc)

            get_param_str(log=True, **run_kwargs)

            run_linear_experiment_grid(
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
