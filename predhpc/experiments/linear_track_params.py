#!/usr/bin/env python3

import argparse
from pathlib import Path
from pprint import pprint
import time

from predhpc import run_manager
from predhpc.experiments import metrics, linear_track
from predhpc.util import params_util, hyper_util, gen_util

NUM_MOVES = 8


def get_save_name(speed_std="high", complex_track=False, target_moved=0):
    """
    get_save_name()

    Get a save name for the hyperparameter search run based on the parameters used.

    Args:
    - speed_std (str, optional): Speed standard deviation to use. Default is "high".
    - complex_track (bool, optional): Whether to run a complex track simulation instead
        of a simple linear track simulation. Default is False.
    - target_moved (int, optional): Number of times to move the target position. If 0,
        the return linear track is run. Default is 0.

    Returns:
    - save_name (str): Save name for the hyperparameter search run, based on the
        parameters used.
    """

    speed_std_str = "_low" if speed_std == "low" else ""
    move_str = "_moved" if target_moved else ""
    lead = "complex" if complex_track else "simple"

    save_name = f"{lead}{move_str}{speed_std_str}"

    return save_name


def get_param_str(speed_std="high", complex_track=False, target_moved=0, log=False):
    """
    get_param_str()

    Get a parameter string based on the parameters used.

    Args:
    - speed_std (str, optional): Speed standard deviation to use. Default is "high".
    - complex_track (bool, optional): Whether to run a complex track simulation instead
        of a simple linear track simulation. Default is False.
    - target_moved (int, optional): Number of times to move the target position. If 0,
        the return linear track is run. Default is 0.
    - log (bool, optional): Whether to log the parameter string. Default is False.

    Returns:
    - param_str (str): Parameter string based on the parameters used.
    """

    param_str = "complex track" if complex_track else "simple track"
    if target_moved:
        param_str = f"{param_str} with {target_moved} target moves"

    param_str = f"{param_str} ({speed_std} speed std)"

    if log:
        print(f"Running {param_str}.")

    return param_str


def get_modes(complex_track=False, target_moved=0):
    """
    get_modes()

    Get the modes for the linear track simulation based on parameters values.

    if complex_track is False:
        If target_moved is 0, the modes are: single, reverse, and back and forth.
        If target_moved is greater than 0, the modes are moved_0, moved_1, ..., moved_n
        in which the target is moved by the same distance each time.
    if complex_track is True:
        The modes are: single, reverse, and back and forth. target_moved must be 0.

    Args:
    - complex_track (bool, optional): Whether to run a complex track simulation instead
        of a simple linear track simulation. Default is False.
    - target_moved (int, optional): Number of times to move the target position. If 0,
        the return linear track is run. Default is 0.

    Returns:
    - modes (list): List of modes to use for the linear track simulation.
    """

    if complex_track:
        if target_moved != 0:
            raise ValueError(
                "target_moved must be 0 if 'complex_track' is True, but is "
                f"{target_moved}."
            )
        modes = ["single", "reverse", "back_and_forth"]
    else:
        if target_moved:
            modes = [f"moved_{i}" for i in range(target_moved)]
        else:
            modes = ["single"]

    return modes


def get_search_space(search_space="full"):
    """
    get_search_space()

    Get the search space for hyperparameter search.

    Args:
    - search_space (str, optional): Search space to use. Default is "full".

    Returns:
    - search_space (dict): Search space for hyperparameter search, with keys and values
        for each pyramidal neuron parameter to search over.
    """

    if isinstance(search_space, str):
        if search_space == "full":
            # in to out, col to row
            search_kwargs = {
                # "soma_regularization_alpha": [2, 5, 5],
                "soma_BTSP_lr": [0.12, 0.24, 5],
                "inhibit_input_filter_tau": [0.2, 0.4, 5],
                "inhibit_weight": [0.8, 1.2, 5],
            }
        else:
            raise ValueError(f"search_space must be 'full', but is {search_space}")
    else:
        search_kwargs = search_space

    search_space = hyper_util.get_search_space(**search_kwargs)

    return search_space


def run_linear_track(
    num_steps=8000,
    skip_runs=1,
    speed_std="high",
    complex_track=False,
    target_moved=0,
    disable_tqdm=False,
    plot=True,
    PF_kwargs=dict(),
    **kwargs,
):
    """
    run_linear_track()

    Run a linear track simulation with the agent optionally in different modes
    (consecutively).

    Args:
    - num_steps (int, optional): Number of steps total to run, for each mode.
        Default is 8000.
    - skip_runs (int, optional): Number of track runs to skip before enabling BTSP.
        Default is 1.
    - speed_std (str, optional): Speed standard deviation to use. Default is "high".
    - complex_track (bool, optional): Whether to run a complex track simulation
        instead of a simple linear track simulation. Default is False.
    - target_moved (int, optional): Number of times to move the target position. If 0,
        the return linear track is run. Default is 0.
    - disable_tqdm (bool, optional): Whether to disable tqdm. Default is False.
    - plot (bool, optional): Whether to generate plots. Default is True.
    - PF_kwargs (dict, optional): Keyword arguments passed to place field analysis
        functions. Default is an empty dictionary.

    Keyword args:
    - **kwargs (dict): Keyword arguments passed to linear_track.get_Pyrs().

    Returns:
    - BTSP_metrics (dict): Dictionary of BTSP metrics, keys and values:
        - "{BTSP_metric}_({mode})": BTSP metric value for specific mode.
    if plot:
    - plot_dict (dict): Dictionary of plots, with keys and values:
        - "spatial_axes_{mode}": Spatial plots for specific mode
        - "time_axes_{mode}": Time plots for specific mode
    """

    Pyrs = linear_track.get_Pyrs(
        scale=params_util.SCALE_LINEAR, speed_std=speed_std, **kwargs
    )

    modes = get_modes(complex_track=complex_track, target_moved=target_moved)

    plot_dict = dict()
    BTSP_metrics = dict()
    for m, mode in enumerate(modes):
        if mode == "reverse":
            Pyrs.Agent.reverse()
        elif "moved" in mode and m != 0:
            Pyrs.Agent.move_target_position(move=-params_util.MOVE_CLOSE * m)

        prev = Pyrs.Agent.get_num_completed_trajectories()

        outputs = run_manager.learn_1D_BTSP(
            Pyrs_or_learner=Pyrs,
            use_Hebbian=False,
            BTSP_on=skip_runs + 1,
            max_num_steps=num_steps,
            record_weights_at_BTSP=False,
            no_logs=disable_tqdm,
            plot=plot,
            reverse=(mode == "back_and_forth"),
        )

        if plot:
            add_str = f"_{mode}" if len(modes) > 1 else ""
            plot_dict[f"spatial_axes{add_str}"] = outputs[1]
            plot_dict[f"time_axes{add_str}"] = outputs[2]

        num_traj_completed = Pyrs.Agent.get_num_completed_trajectories() - prev
        if num_traj_completed < skip_runs + 2:
            raise RuntimeError(f"Only {num_traj_completed} trajectories completed.")

        t_start = int(Pyrs.Agent.trajectory_df.loc[skip_runs + prev, "start_time"])
        mode_BTSP_metrics = metrics.compute_BTSP_metrics(
            Pyrs, t_start=t_start, **PF_kwargs
        )

        add_str = f"_({mode})" if len(modes) > 1 else ""
        for key, value in mode_BTSP_metrics.items():
            BTSP_metrics[f"{key}{add_str}"] = value

    if plot:
        return BTSP_metrics, plot_dict
    else:
        return BTSP_metrics


def run_hyperparameter_search(
    speed_std="high",
    complex_track=False,
    target_moved=0,
    direc=None,
    num_CPUs=4,
    num_repeats=4,
    search_space="full",
    disable_tqdm=True,
    plot=False,
    debug=False,
    **kwargs,
):
    """
    run_hyperparameter_search()

    Run a hyperparameter search for a linear track simulation.

    Args:
    - speed_std (str, optional): Speed standard deviation to use. Default is "high".
    - complex_track (bool, optional): Whether to run a complex track simulation instead
        of a simple linear track simulation. Default is False.
    - target_moved (int, optional): Number of times to move the target position. If 0,
        the return linear track is run. Default is 0.
    - direc (str, optional): Directory to save results in. If None, a default
        directory is used (see hyper_util.get_save_directory()). Default is None.
    - num_CPUs (int, optional): Number of CPUs to run search across. Default is 4.
    - num_repeats (int, optional): Number of repeats for each hyperparameter set.
        Default is 4.
    - search_space (str, optional): Search space to use. Default is "full".
    - disable_tqdm (bool, optional): Whether to disable tqdm. Default is True.
    - plot (bool, optional): Whether to generate plots. Default is False.
    - debug (bool, optional): Whether to run in debug mode. Default is False.

    Keyword args:
    - **kwargs (dict): Keyword arguments passed to linear_track.get_Pyrs().
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

        kwargs_use = kwargs.copy()

        kwargs_use.update(config)
        if "soma_regularization_alpha" in kwargs_use.keys():
            kwargs_use["soma_normalize_weights_divisively"] = True

        output_dict = run_linear_track(
            speed_std=speed_std,
            complex_track=complex_track,
            target_moved=target_moved,
            disable_tqdm=disable_tqdm,
            plot=plot,
            **kwargs_use,
        )

        return output_dict

    search_space = get_search_space(search_space=search_space)
    save_name = get_save_name(speed_std, complex_track, target_moved=target_moved)

    hyper_util.run_hyperparameter_search(
        objective,
        search_space,
        direc=direc,
        save_name=save_name,
        num_CPUs=num_CPUs,
        num_repeats=num_repeats,
        debug=debug,
    )


def yield_kwargs(
    low_speed_std=False, complex_track=False, target_moved=False, cycle_all=False
):
    """
    yield_kwargs()

    Yield keyword arguments for linear track simulations.

    Args:
    - low_speed_std (bool, optional): Whether to use low speed standard deviation.
        Default is False.
    - complex_track (bool, optional): Whether to run a complex track simulation instead
        of a simple linear track simulation. Default is False.
    - target_moved (bool, optional): Whether to move the target position, if running
        return track simulation. Default is False.
    - cycle_all (bool, optional): Whether to cycle through all possible combinations of
        keyword arguments. Default is False.

    Yields:
    - kwargs (dict): Keyword arguments for linear track simulation.
    """

    if cycle_all:
        for speed_std in ["low", "high"]:
            for track in ["simple", "move", "complex"]:
                kwargs = {
                    "speed_std": speed_std,
                    "complex_track": track == "complex",
                    "target_moved": NUM_MOVES if track == "move" else 0,
                }
                yield kwargs

    else:
        if target_moved and complex_track:
            raise ValueError("target_moved must be False for complex linear track.")

        kwargs = {
            "speed_std": "low" if low_speed_std else "high",
            "complex_track": complex_track,
            "target_moved": target_moved,
        }
        yield kwargs


def get_args():
    """
    get_args()

    Get command line arguments for linear track simulations.

    Returns:
    - args (argparse.Namespace): Parsed command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_steps", type=int, default=8000)
    parser.add_argument("--low_speed_std", action="store_true")
    parser.add_argument("--complex_track", action="store_true")
    parser.add_argument("--target_moved", type=int, default=0)
    parser.add_argument("--direc", type=Path, default=None)
    parser.add_argument("--num_CPUs", type=int, default=2)
    parser.add_argument("--num_repeats", type=int, default=4)
    parser.add_argument("--hyperparameter_search", action="store_true")
    parser.add_argument("--cycle_all", action="store_true")
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
        all_run_kwargs = yield_kwargs(
            low_speed_std=args.low_speed_std,
            complex_track=args.complex_track,
            target_moved=args.target_moved,
            cycle_all=args.cycle_all,
        )

        for run_kwargs in all_run_kwargs:
            start_time = time.perf_counter()

            get_param_str(log=True, **run_kwargs)

            if args.hyperparameter_search:
                run_hyperparameter_search(
                    direc=args.direc,
                    num_CPUs=args.num_CPUs,
                    num_repeats=args.num_repeats,
                    num_steps=args.num_steps,
                    debug=args.debug,
                    **run_kwargs,
                )
            else:
                BTSP_metrics = run_linear_track(
                    num_steps=args.num_steps, plot=False, **run_kwargs
                )
                print("\nBTSP metrics:")
                pprint(BTSP_metrics)

            gen_util.get_duration_str(start_time, log=True)


if __name__ == "__main__":
    main()
