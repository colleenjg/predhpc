#!/usr/bin/env python3

import argparse
from pathlib import Path
import itertools
import multiprocessing
import pickle as pkl
from joblib import Parallel, delayed

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from predhpc.util import signal_util, ext_util, plot_util, params_util, gen_util

OUTER_LOSS_WEIGHT = 0.2

TARGET_ROOT_DICT = {
    "pre_outer": -6,
    "pre_center": -2.5,
    "post_center": 1.5,
    "post_outer": 6,
}


def get_save_directory(direc=None, kernel=False):
    """
    get_save_directory()

    Obtain, and create if necessary, the save directory for hyperparameter search
    results.

    Args:
    - direc (str or Path): Directory to save results in. If None, a default
        directory is used (e.g., ../results/hyperparameter_search). Default is None.
    - kernel (bool, optional): Whether the directory is for kernel search results.
        Default is False.

    Returns:
    - direc (Path): Directory to save results in.
    """

    if direc is None:
        sub_direc = "kernel_search" if kernel else "hyperparameter_search"
        direc = Path("..", "results", sub_direc)
    direc = Path(direc)
    direc.mkdir(parents=True, exist_ok=True)

    return direc


def replot_from_csvs(replotting_path):
    """
    replot_from_csvs(replotting_path)

    Replot metrics from CSV files in the specified directory.

    Args:
    - replotting_path (Path): Path to the directory containing CSV files to replot.
    """

    replotting_path = Path(replotting_path)

    if replotting_path.is_dir():
        # get csvs using glob
        csvs = list(replotting_path.glob("**/*.csv"))
        replot_str = f"{len(csvs)} csvs"
    else:
        csvs = [replotting_path]
        replot_str = replotting_path

    print(f"Replotting from {replot_str}.")
    for csv_path in tqdm(csvs):
        df = pd.read_csv(csv_path)
        direc = csv_path.parent
        save_name = csv_path.stem
        plot_metrics_by_parameters(df, direc=direc, name_str=save_name)


def get_search_space(**kwargs):
    """
    get_search_space()

    Get search space for hyperparameter search.

    Keyword args:
    - kwargs (dict): Keyword arguments specifying the search space parameters.

    Returns:
    - search_space (dict): Search space dictionary specifying parameters to search
        through.
    """

    from ray import tune

    search_space = dict()
    for parameter, values in kwargs.items():
        if len(values) == 3 and isinstance(values[-1], int):
            values = np.linspace(*values)
        else:
            values = np.asarray(values)

        search_space[parameter] = tune.grid_search(values)

    return search_space


def check_parameters(
    side=None,
    log=False,
    **kwargs,
):
    """
    check_parameters()

    Checks that parameter values defining the kernel are valid.

    Args:
    - side (str): The side for which the parameters are used ("pre" or "post"). Used
        for debugging message. Default is None.
    - log (bool): If True, and certain parameters are not valid, a descriptive
        message will be logged.

    Keyword args:
    - **kwargs: Values defining the kernel. See
        params_util.get_default_BTSP_filter_param_dict().

    Returns:
    - valid (bool): Whether the parameters are valid.
    """

    kernel_dict = params_util.get_default_BTSP_filter_param_dict(incl_BTSP_str=False)
    for key, item in kwargs.items():
        if key not in kernel_dict.keys():
            raise KeyError(f"Kernel key not recognized: {key}.")
        kernel_dict[key] = item

    valid = True

    for side in ["pre", "post"]:
        filter_tau_pos = kernel_dict[f"{side}_filter_tau_pos"]
        filter_tau_neg = kernel_dict[f"{side}_filter_tau_neg"]
        neg_weight = kernel_dict[f"{side}_neg_weight"]

        if filter_tau_neg < filter_tau_pos:
            if log:
                print(
                    f"Tau for {side} negative filter component ({filter_tau_neg}) "
                    f"is smaller than tau for {side} positive filter component "
                    f"({filter_tau_pos})."
                )
            valid = False

        if neg_weight == 1.0:
            if log:
                print(
                    f"The negative weight for {side} is 1. This will lead to "
                    "a division by 0."
                )
            valid = False

        adj_neg_weight = signal_util.get_norm_adj_neg_weight(
            filter_tau_pos=filter_tau_pos,
            filter_tau_neg=filter_tau_neg,
            neg_weight=neg_weight,
        )

        if adj_neg_weight == 1.0:
            if log:
                print(
                    f"The adjusted negative weight for {side} is 1. This will lead to "
                    "a division by 0."
                )
            valid = False

    return valid


def check_exp(exp, align_pt=None, log=False):
    """
    check_exp(exp)

    Args:
    - exp (1D np.ndarray): Exponential to check.
    - align_pt (int or None): Point at which the pre and post exponential components
        align. If None, the peak of the exponential is used. Default is None.
    - log (bool): If True, and the exponential is not valid, a descriptive
        message will be logged.

    Returns:
    - valid (bool): Whether the exponential is valid.
    """

    valid = True

    if align_pt is None:
        align_pt = np.argmax(exp)

    if exp.min() > 0:
        if log:
            print("Signal does not go below 0.")
        valid = False

    if exp.sum() < 0:
        if log:
            print(f"Integral is below 0: {exp.sum():.4f}.")
        valid = False

    for side in ["pre", "post"]:
        min_neg = get_min_neg_value(side=side)
        sub_exp = exp[align_pt:] if side == "post" else exp[: align_pt + 1]
        if -sub_exp.min() < min_neg:
            if log:
                print(
                    f"Target minimum negative value for {side} ({min_neg:.4f}) not "
                    f"reached: {-sub_exp.min():.4f}."
                )
            valid = False

    return valid


def get_parameters_from_df(df, parameters=None):
    """
    get_parameters_from_df(df)

    Obtain list of parameters, either by comparing to the hyperparameter search
    dataframe or extracting them from the dataframe. Parameters are columns starting
    with 'config/'.

    Args:
    - df (pd.DataFrame): Dataframe in which hyperparameter search results are recorded.
    - parameters (list of str, optional): List of parameters to check against the
        dataframe. If None, they are extracted from the dataframe
        (columns starting with 'config/'). Default is None.

    Returns:
    - parameters (list of str): List of parameters.
    """

    if parameters is None:
        parameters = [col for col in df.columns if col.startswith("config/")]
    else:
        columns = list()
        for parameter in parameters:
            if not parameter.startswith("config/"):
                parameter = f"config/{parameter}"
            if parameter not in df.columns:
                raise ValueError(f"{parameter} not in dataframe.")
            columns.append(parameter)
        parameters = columns

    return parameters


def get_num_repeats_from_df(df, parameters=None):
    """
    get_num_repeats_from_df(df)

    Obtain the number of repeats from the hyperparameter search dataframe.

    Args:
    - df (pd.DataFrame): Dataframe in which hyperparameter search results are recorded.
    - parameters (list of str, optional): List of parameters to check against the
        dataframe. If None, they are extracted from the dataframe. Default is None.

    Returns:
    - num_repeats (int): Number of repeats.
    """

    parameters = get_parameters_from_df(df, parameters=parameters)

    num_repeats = 1
    for _, grp in df.groupby(parameters):
        num_repeats = int(max(len(grp), num_repeats))

    return num_repeats


def get_metric_plot_vmin_vmax(metric="num_BTSP_events"):
    """
    get_metric_plot_vmin_vmax()

    Obtain vmin and vmax for plotting a specific metric.

    Args:
    - metric (str, optional): Metric to plot. Default is "num_BTSP_events".

    Returns:
    - vmin (float, optional): Minimum value for plotting the metric.
    - vmax (float, optional): Maximum value for plotting the metric.
    """

    vmin, vmax = None, None
    if metric.startswith("num"):
        vmin = 1
    elif "time" in metric or "ramp" in metric:
        vmin = 0
    elif "relative_position" in metric:
        vmin, vmax = -1, 1
    elif "position" in metric:
        vmin, vmax = 0, 1
    return vmin, vmax


def get_metric_plot_title(metric="num_BTSP_events", median=None):
    """
    get_metric_plot_title()

    Obtain the title for plotting a metric.

    Args:
    - metric (str, optional): Metric to plot. Default is "num_BTSP_events".
    - median (float, optional): Median value for the metric. Default is None.

    Returns:
    - title (str): Title for the plot.
    """

    title_parts = metric.replace("_", " ").split(" (")
    if "ratio" in metric:
        title_parts[0] = f"{title_parts[0]} (log)"
    title = "\n (".join(title_parts)
    if median is not None:
        title = f"{title}\n(median={median:.2f})"
    return title


def get_values_coords_and_labels(df, parameters=None):
    """
    get_values_coords_and_labels(df)

    Obtain values, coordinates, and labels for plotting metrics by parameters.

    Args:
    - df (pd.DataFrame): Dataframe in which hyperparameter search results are recorded.
    - parameters (list of str, optional): List of parameters to plot the metric by.
        If None, they are extracted from the dataframe. Default is None.

    Returns:
    - values_dict (dict): Dictionary of parameter values, where the keys and values are
        the parameter names and list of parameter values, respectively.
    - coords (list of int): List of coordinates for plotting for creating the metric
        array.
    - param_labels (list of list of str): List of parameter labels for the plot.
    """

    num_repeats = get_num_repeats_from_df(df)
    parameters = get_parameters_from_df(df, parameters=parameters)

    values_dict = dict()
    coords = [1, 1]
    param_labels = [[f"repeat ({num_repeats})"], list()]
    for p, parameter in enumerate(parameters):
        values_dict[parameter] = list(df[parameter].unique())
        idx = (p + 1) % 2
        coords[idx] *= len(values_dict[parameter])
        param_name = parameter.replace("config/", "").replace("_", " ")
        param_vals = [f"{np.around(val, 10)}" for val in values_dict[parameter]]
        if idx != 0:  # reverse only the labels (to better match plotted order)
            param_vals = param_vals[::-1]
        if len(param_vals) < 6:
            param_vals = ", ".join(param_vals)
        else:
            param_vals = f"{param_vals[0]} to {param_vals[-1]} [{len(param_vals)}]"
        param_label = f"{param_name} ({param_vals})"
        param_labels[idx].insert(0, param_label)
    coords[0] *= num_repeats

    return values_dict, coords, param_labels


def get_metric_scatter_data(df, parameters=None, metric="num_BTSP_events"):
    """
    get_metric_scatter_data(df)

    Obtain data for plotting a metric as a scatter plot, organized by each parameter.

    Args:
    - df (pd.DataFrame): Dataframe in which hyperparameter search results are recorded.
    - parameters (list of str, optional): List of parameters to plot the metric by.
        If None, they are extracted from the dataframe. Default is None.
    - metric (str, optional): Metric to plot. Must correspond to a dataframe column
        (column starting with 'metric/'). Default is "num_BTSP_events".

    Returns:
    - metric_array (1D np.ndarray): Array of metric values.
    - param_value_dict (dict): Dictionary of parameter values, where the keys are
        parameter names and the values are arrays of parameter values for each value
        in metric_array.
    """

    parameters = get_parameters_from_df(df, parameters=parameters)

    metric_col = f"metric/{metric}"
    if metric_col not in df.columns:
        raise ValueError(f"{metric_col} not in dataframe.")

    metric_array = df[metric_col].to_numpy()

    param_value_dict = dict()
    for parameter in parameters:
        if parameter not in df.columns:
            raise ValueError(f"{parameter} not in dataframe.")
        param_value_dict[parameter.replace("config/", "")] = df[parameter].to_numpy()

    return metric_array, param_value_dict


def get_metric_array(
    df, parameters=None, metric="num_BTSP_events", ascending=True, scatter_data=False
):
    """
    get_metric_array(df)

    Obtain an array of metric values, organized by each parameter.

    Args:
    - df (pd.DataFrame): Dataframe in which hyperparameter search results are recorded.
    - parameters (list of str, optional): List of parameters to plot the metric by.
        If None, they are extracted from the dataframe. Default is None.
    - metric (str, optional): Metric to plot. Default is "num_BTSP_events".
    - ascending (bool, optional): Whether to sort the parameters in ascending order,
        instead of descending order. Default is True.
    - scatter_data (bool, optional): Whether to return data for a scatter plot
        instead of a colormap. Only possible if there is only one parameter.
        Default is False.

    Returns:
    - metric_array (2D np.ndarray): Array of metric values, organized by each parameter.
    - param_labels (list of list of str): List of parameter labels for the plot.
    """

    values_dict, coords, param_labels = get_values_coords_and_labels(df, parameters)

    metric_col = f"metric/{metric}"
    if metric_col not in df.columns:
        raise ValueError(f"{metric_col} not in dataframe.")

    num_repeats = get_num_repeats_from_df(df, parameters=parameters)
    parameters = get_parameters_from_df(df, parameters=parameters)

    metric_array = np.full(coords, np.nan)
    for grp_vals, grp in df.groupby(parameters):
        x, y = 0, 0
        x_block, y_block = num_repeats, 1
        for g, val in enumerate(grp_vals):
            values = values_dict[parameters[g]]
            val_idx = values.index(val)
            if g % 2:  # x val
                x += val_idx * x_block
                x_block *= len(values)
            else:  # y val
                y += val_idx * y_block
                y_block *= len(values)

        vals = np.sort(grp[metric_col].to_numpy())
        if not ascending:
            vals = vals[::-1]
        metric_array[x : x + len(vals), y] = np.sort(vals)

    return metric_array, param_labels


def plot_metric_by_parameters(
    df,
    sub_ax=None,
    parameters=None,
    metric="num_BTSP_events",
    ascending=True,
):
    """
    plot_metric_by_parameters(df)

    Plot a metric by parameters.

    Args:
    - df (pd.DataFrame): Dataframe in which hyperparameter search results are recorded.
    - sub_ax (matplotlib.axes.Axes, optional): Subplot to plot in. Default is None.
    - parameters (list of str, optional): List of parameters to plot the metric by.
        If None, they are extracted from the dataframe
        (columns starting with 'config/'). Default is None.
    - metric (str, optional): Metric to plot. Must correspond to a dataframe column
        (column starting with 'metric/'). Default is "num_BTSP_events".
    - ascending (bool, optional): Whether to sort the parameters in ascending order,
        instead of descending order. Default is True.

    Returns:
    - metric_array (2D np.ndarray): Array of metric values, organized by each parameter.
    - im (matplotlib.image.AxesImage): Image from the plot. None if a scatter plot is
        created.
    """

    parameters = get_parameters_from_df(df, parameters=parameters)
    single_parameter = len(parameters) == 1

    if single_parameter:
        metric_array, param_dict = get_metric_scatter_data(
            df, parameters=parameters, metric=metric
        )
    else:
        metric_array, param_labels = get_metric_array(
            df,
            parameters=parameters,
            metric=metric,
            ascending=ascending,
        )

    # log transform metric, if ratio
    if "ratio" in metric:
        metric_array = np.log(metric_array)

    # get title before setting 0s to nan for plotting
    title = get_metric_plot_title(metric, median=np.nanmedian(metric_array))

    # start plotting
    if sub_ax is None:
        _, sub_ax = plt.subplots()

    sub_ax.set_title(title)

    vmin, vmax = get_metric_plot_vmin_vmax(metric)

    im = None
    if len(parameters) == 1:
        param_keys = list(param_dict.keys())
        if len(param_keys) != 1:
            raise RuntimeError(
                f"Only one parameter expected, but found {len(param_keys)}."
            )
        # plot a scatter plot
        param_key = param_keys[0]
        param_values = np.asarray(param_dict[param_key])
        mask = np.isfinite(metric_array)

        if mask.sum():
            sub_ax.scatter(
                param_values[mask], metric_array[mask], s=10, c="k", alpha=0.3
            )
            if vmin is not None and vmin < metric_array[mask].min():
                sub_ax.set_ylim(vmin, None)
            if vmax is not None and vmax > metric_array[mask].max():
                sub_ax.set_ylim(None, vmax)
            if vmin is not None or vmax is not None:
                plot_util.pad_axis(sub_ax, axis="y")

        if param_key == "target_moved":
            sub_ax.axvline(0, ls="dashed", color="k", zorder=-5)

        param_str = f"{param_key[0].upper()}{param_key[1:].replace('_', ' ')}"
        sub_ax.set_xlabel(param_str)
        sub_ax.spines[["top", "right"]].set_visible(False)

    else:
        if metric.startswith("num"):
            zero_mask = metric_array == 0
            metric_array[zero_mask] = np.nan  # for plotting

        im = sub_ax.imshow(metric_array.T, cmap="viridis", vmin=vmin, vmax=vmax)

        sub_ax.figure.colorbar(im, ax=sub_ax, orientation="vertical")
        sub_ax.set_xlabel(" x\n".join(param_labels[0]))
        sub_ax.set_xticks([])
        sub_ax.set_ylabel(" x\n".join(param_labels[1]))
        sub_ax.set_yticks([])

        # add vertical lines for each repeat
        num_repeats = get_num_repeats_from_df(df, parameters=parameters)
        num_x_lines = int(len(metric_array) / num_repeats) + 1
        for x in np.linspace(0, len(metric_array), num_x_lines)[1:-1]:
            sub_ax.axvline(x - 0.5, lw=1, color="k")

        # add horizontal lines for each first parameter value
        first_parameter = get_parameters_from_df(df, parameters=parameters)[0]
        num_values = len(list(df[first_parameter].unique()))
        num_y_lines = int(metric_array.shape[1] / num_values) + 1
        for y in np.linspace(0, metric_array.shape[1], num_y_lines)[1:-1]:
            sub_ax.axhline(y - 0.5, lw=1, color="k")

        if metric.startswith("num") and zero_mask.sum():
            metric_array[zero_mask] = 0

    return metric_array, im


def plot_metrics_by_parameters(df, parameters=None, direc=None, name_str=None):
    """
    plot_metrics_by_parameters(df)

    Plot metrics organized by parameters, with one metric in each subplot.

    Args:
    - df (pd.DataFrame): Dataframe in which hyperparameter search results are recorded.
    - parameters (list of str, optional): List of parameters to plot the metric by.
        If None, they are extracted from the dataframe
        (columns starting with 'config/'). Default is None.
    - direc (str or Path, optional): Directory to save results in. If None,
        the current working directory is used. Default is None.
    - name_str (str, optional): Name of the file in which to save the plot.
        Default is None.

    Returns:
    - axes (2D np.ndarray): Array of axes to plot on. There is one subplot per metric.
    """

    metrics = [
        col.replace("metric/", "") for col in df.columns if col.startswith("metric/")
    ]

    direc = get_save_directory(direc)

    if name_str is None:
        name_str = "metrics"

    if len(metrics) == 0:
        raise RuntimeError("No metrics found in dataframe.")

    if len(metrics) < 24:
        ncols = min(4, len(metrics))
    else:
        ncols = 8
    nrows = int(np.ceil(len(metrics) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=[ncols * 6, nrows * 6],
        squeeze=False,
        gridspec_kw={"hspace": 0.3},
    )
    for a, sub_ax in enumerate(axes.ravel()):
        if a < len(metrics):
            plot_metric_by_parameters(df, sub_ax, parameters, metric=metrics[a])
        else:
            sub_ax.axis("off")

    for suffix in ["png", "svg"]:
        fig_path = Path(direc, f"{name_str}.{suffix}")
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    return axes


def run_hyperparameter_search(
    objective,
    search_space,
    direc=None,
    save_name="hyperparameter_search",
    num_CPUs=4,
    num_repeats=4,
    debug=False,
    **plot_kwargs,
):
    """
    run_hyperparameter_search(objective, search_space)

    Run a hyperparameter search using Ray Tune.

    Args:
    - objective (function): Objective function to optimize.
    - search_space (dict): Dictionary of hyperparameters to search over.
    - direc (str or Path, optional): Directory to save results in.
        Default is None.
    - save_name (str, optional): Name of the file in which to save hyperparameter
        search results. Default is "hyperparameter_search".
    - num_CPUs (int, optional): Number of CPUs to use for the search. Default is 4.
    - num_repeats (int, optional): Number of repeats to use in grid search.
        Default is 4.
    - debug (bool, optional): Whether to log to Ray Tune driver. Default is False.

    Keyword args:
    - **plot_kwargs: Keyword arguments passed to plotting passed to
        hyper_util.plot_metrics_by_parameters().

    Returns:
    - tuner (ray.tune.Tuner): Fitted Ray Tune tuner object.
    """

    import ray
    from ray import tune, air

    ray.init(num_cpus=num_CPUs, log_to_driver=debug)

    # calculate number of runs (for grid search)
    num_combs = 1
    for hyperparam in search_space.values():
        num_combs *= len(hyperparam["grid_search"])
    repeat_str = "repeat" if num_repeats == 1 else "repeats"
    CPU_str = "CPU" if num_CPUs == 1 else "CPUs"
    print(
        f"\nRunning {num_combs} hyperparameter combinations ({num_repeats} "
        f"{repeat_str} => {num_repeats * num_combs} total runs) using {num_CPUs} "
        f"{CPU_str}.\n"
    )

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            num_samples=num_repeats,  # number of repeats, for grid searches
        ),
        run_config=air.RunConfig(verbose=0),
        param_space=search_space,
    )

    results = tuner.fit()
    ray.shutdown()

    df = results.get_dataframe()

    direc = get_save_directory(direc)
    date_time_str = gen_util.get_date_time_str()

    save_name = f"{save_name}_{date_time_str}"
    save_path = Path(direc, f"{save_name}.csv")
    df.to_csv(save_path)

    parameters = list(search_space.keys())
    plot_metrics_by_parameters(
        df,
        parameters,
        direc=direc,
        name_str=save_name,
        **plot_kwargs,
    )

    return tuner


def get_target_root_dict(side="both"):
    """
    get_target_root_dict()

    Obtain the target root dictionary for the kernel values.

    Args:
    - side (str, optional): Side of the target kernel values to use. Default is "both".

    Returns:
    - target_root_dict (dict): Dictionary of target root values.
    """

    target_root_dict = TARGET_ROOT_DICT.copy()

    if side == "pre":
        target_root_dict["post_center"] = -target_root_dict["pre_center"]
        target_root_dict["post_outer"] = -target_root_dict["pre_outer"]
    elif side == "post":
        target_root_dict["pre_center"] = -target_root_dict["post_center"]
        target_root_dict["pre_outer"] = -target_root_dict["post_outer"]
    elif side != "both":
        raise ValueError(f"Invalid side: {side}")

    return target_root_dict


def get_kernel_loss(
    val_dict,
    use_mse=True,
    outer_loss_weight=OUTER_LOSS_WEIGHT,
    roots=False,
    side="both",
):
    """
    get_kernel_loss(val_dict)

    Obtain the loss for a kernel by comparing actual values compared to target values.
    If roots are provided, they are compared to the target root values recorded.
    Otherwise, all values are compared to 0.

    Args:
    - val_dict (dict): Dictionary of root values.
    - mse (bool, optional): Whether to calculate the mean squared error.
        Default is True.
    - outer_loss_weight (float, optional): Weight by which to multiply the loss value
        for outer roots compared to the loss value for central roots.
        Default is OUTER_LOSS_WEIGHT.
    - roots (bool, optional): Whether the values for which to compute the loss are
        root values. Default is False.
    - side (str, optional): Side of the target root values to compare to, if roots is
        True. Default is "both".

    Returns:
    - loss (float): Loss for the kernel values.
    """

    if roots:
        target_dict = get_target_root_dict(side=side)
    else:
        target_dict = {key: 0 for key in val_dict.keys()}

    comps = np.zeros(len(target_dict.keys()))
    for i, (key, targ_val) in enumerate(target_dict.items()):
        if key not in val_dict.keys():
            raise KeyError(f"'{key}' not found.")
        val = val_dict[key]
        comps[i] = val - targ_val
        if roots and targ_val != 0:
            comps[i] /= targ_val
        if use_mse:
            comps[i] **= 2
        if "outer" in key:
            comps[i] *= outer_loss_weight

    loss = np.mean(np.absolute(comps))

    return loss


def get_root_dict(
    exp=None,
    align_pt=None,
    dt=0.03,
    near_zero=1e-4,
    debug=False,
    use_mse=True,
    outer_loss_weight=OUTER_LOSS_WEIGHT,
    **kwargs,
):
    """
    get_root_dict()

    Obtain the roots of the kernel values, based on the target kernel values.

    Args:
    - exp (1D np.ndarray, optional): Exponential to check. If None, it is calculated
        using signal_util.get_summed_exp() and kwargs. Default is None.
    - align_pt (int or None): Point at which the pre and post exponential components
        align. If None and exp is provided, the peak of the exponential is used.
        Default is None.
    - dt (float, optional): Time step. Default is 0.03.
    - near_zero (float, optional): Proportion of maximum value to use to identify outer
        near zero values. Default is 1e-4.
    - debug (bool, optional): Whether to print debug information. Default is False.
    - use_mse (bool, optional): Whether to use mean squared error for the loss.
        Default is False.
    - outer_loss_weight (float, optional): Weight by which to multiply the loss value
        for outer roots compared to the loss value for central roots.
        Default is OUTER_LOSS_WEIGHT.

    Keyword args:
    - **kwargs: Values defining the kernel. See signal_util.get_summed_exp().

    Returns:
    - root_dict (dict): Dictionary of kernel roots.
    """

    if exp is None:
        exp, exp_align_pt = signal_util.get_summed_exp(dt=dt, **kwargs)
        if align_pt is None:
            align_pt = exp_align_pt

    if align_pt is None:
        align_pt = np.argmax(exp)

    t = (np.arange(len(exp)) - align_pt) * dt

    root_dict = dict()
    for root_type in ["center", "outer"]:
        use_exp = exp
        if root_type == "outer":
            use_exp = exp + near_zero * exp.max()
        for side in ["pre", "post"]:
            if side == "pre":
                t_idxs = np.arange(align_pt + 1)
                if root_type == "center":
                    t_idxs = t_idxs[::-1]
            else:
                t_idxs = np.arange(align_pt, len(exp))
                if root_type == "outer":
                    t_idxs = t_idxs[::-1]

            if use_exp[t_idxs[0]] <= 0:
                break

            for i in range(1, len(t_idxs)):
                if np.sign(use_exp[t_idxs[i]]) == -1:
                    ts = (t[t_idxs[i - 1]], t[t_idxs[i]])
                    root = brentq(lambda x: np.interp(x, t, use_exp), *ts)
                    root_dict[f"{side}_{root_type}"] = root
                    break

    if len(root_dict) != 4:
        if debug:
            root_str = get_root_str(root_dict)
            print(f"Only {len(root_dict)}/4 roots found:\n  {root_str}")
        return np.inf

    if debug:
        root_loss = get_kernel_loss(
            root_dict,
            roots=True,
            use_mse=use_mse,
            outer_loss_weight=outer_loss_weight,
        )
        root_str = get_root_str(root_dict)

        print(f"Roots:\n  {root_str}")
        print(f"Root loss: {root_loss:.4f}")

    return root_dict


def get_values_at_target_roots(
    exp=None,
    align_pt=None,
    dt=0.03,
    **kwargs,
):
    """
    get_values_at_target_roots()

    Obtain the values at the target roots of the kernel.

    Args:
    - exp (1D np.ndarray, optional): Exponential to check. If None, it is calculated
        using signal_util.get_summed_exp() and kwargs. Default is None.
    - align_pt (int or None, optional): Point at which the pre and post exponential
        components align. If None and exp is provided, the peak of the exponential is
        used. Default is None.
    - dt (float, optional): Time step. Default is 0.03.

    Keyword args:
    - **kwargs: Values defining the kernel. See signal_util.get_summed_exp().

    Returns:
    - val_dict (dict): Dictionary of kernel values at target roots.
    """

    if exp is None:
        exp, exp_align_pt = signal_util.get_summed_exp(dt=dt, **kwargs)
        if align_pt is None:
            align_pt = exp_align_pt

    if align_pt is None:
        align_pt = np.argmax(exp)

    t = (np.arange(len(exp)) - align_pt) * dt

    target_root_dict = get_target_root_dict()

    val_dict = dict()
    for name, target_t in target_root_dict.items():
        if target_t in t:
            val_dict[name] = exp[np.where(t == target_t)[0][0]]
        else:
            spline = CubicSpline(t, exp)
            val_dict[name] = spline(target_t)

    return val_dict


def get_root_str(root_dict, link="\n  "):
    """
    get_root_str(root_dict)

    Obtain a string of the root values.

    Args:
    - root_dict (dict): Dictionary of root values.
    - link (str, optional): Link between root values. Default is "\n  ".

    Returns:
    - root_str (str): String of the root values.
    """

    names = [key.replace("_", " ").capitalize() for key in root_dict.keys()]
    values = [val for val in root_dict.values()]
    order = np.argsort(values)

    root_strs = list()
    for i in order:
        root_strs.append(f"{names[i]}: {values[i]:.4f}")
    root_str = link.join(root_strs)

    return root_str


def get_complete_kernel_dict(**kwargs):
    """
    get_complete_kernel_dict()

    Obtain the complete the kernel dictionary with the negative delta values.

    Keyword args:
    - **kwargs: Values defining the kernel. See
        params_util.get_default_BTSP_filter_param_dict().

    Returns:
    - kernel_dict (dict): Completed dictionary of kernel values.
    """

    kernel_dict = params_util.get_default_BTSP_filter_param_dict(
        incl_BTSP_str=False, neg_delta=True
    )
    kernel_dict.update(kwargs)

    for side in ["pre", "post"]:
        delta = kernel_dict.pop(f"{side}_neg_delta")
        pos = kernel_dict[f"{side}_filter_tau_pos"]
        kernel_dict[f"{side}_filter_tau_neg"] = pos + delta

    return kernel_dict


def evaluate_kernel(
    dt=0.03,
    sigma_in_steps=None,
    near_zero=1e-4,
    debug=False,
    side="both",
    use_mse=True,
    **kwargs,
):
    """
    evaluate_kernel()

    Evaluate the kernel generated by the provided kernel parameters. In particular, the
    inner roots and outer near roots.

    Args:
    - dt (float, optional): Time step for Gaussian filter. Default is 0.03.
    - sigma_in_steps (float, optional): Sigma for Gaussian filter in steps.
        Default is None.
    - near_zero (float, optional): Proportion of maximum value to use to identify outer
        near zero values. Default is 1e-4.
    - debug (bool, optional): Whether to print debug information. Default is False.
    - use_mse (bool, optional): Whether to use mean squared error for the loss.
        Default is False.
    - ignore_warnings (list of str, optional): List of RuntimeWarning messages to
        ignore. Default is KERNEL_IGNORE_WARNINGS.

    Keyword args:
    - **kwargs: Values defining the kernel. See
        params_util.get_default_BTSP_filter_param_dict().

    Returns:
    - loss (float): Loss for the kernel values.
    """

    kernel_dict = get_complete_kernel_dict(**kwargs)

    valid = check_parameters(side=side, log=debug, **kernel_dict)
    if not valid:
        return np.inf

    full_exp, align_pt = signal_util.get_summed_exp(
        dt=dt, sigma_in_steps=sigma_in_steps, **kernel_dict
    )

    valid = check_exp(full_exp, align_pt, log=debug)
    if not valid:
        return np.inf

    val_dict = get_values_at_target_roots(
        full_exp, align_pt, dt=dt, debug=debug, use_mse=use_mse
    )

    loss = get_kernel_loss(val_dict, roots=False, use_mse=use_mse)

    if debug:
        root_str = get_root_str(val_dict)
        print(f"Values at roots:\n  {root_str}")
        print(f"Loss: {loss:.4f}\n")
        get_root_dict(
            full_exp, align_pt, dt=dt, near_zero=near_zero, debug=debug, use_mse=use_mse
        )

    return loss


def get_best_index(param_grid, best_indices):
    """
    get_best_index(param_grid, best_indices)

    Obtain the best index from the grid search.

    Args:
    - param_grid (list): List of parameter values.
    - best_indices (list): List of best indices.

    Returns:
    - best_index (int): Index of best parameter values.
    """

    if len(best_indices) > 1:
        param_grid = np.asarray(param_grid)

        selected = [None, None, None]
        while any([val is None for val in selected]):
            all_values = param_grid[best_indices].T
            dim, val, keep = None, None, None
            max_count = 0
            for d, dim_values in enumerate(all_values):
                if selected[d] is None:
                    unique, counts = np.unique(dim_values, return_counts=True)
                    if counts.max() > max_count:
                        dim, val = d, unique[np.argmax(counts)]
                        keep = np.where(dim_values == val)[0]
                        max_count = counts.max()

            selected[dim] = val
            best_indices = best_indices[keep]

    best_index = best_indices[0]

    return best_index


def run_kernel_gridsearch(
    kernel_search_dict,
    log=False,
    plot_results=False,
    debug=False,
    **kwargs,
):
    """
    run_kernel_gridsearch()

    Run a grid search for the kernel values.

    Args:
    - kernel_search_dict (dict): Dictionary of kernel parameters to search over.
    - log (bool, optional): Whether to log the results. Default is False.
    - plot_results (bool, optional): Whether to plot the results. Default is False.
    - debug (bool, optional): If True, search is done in debugging mode.

    Returns:
    - result_dict (dict): Dictionary of gridsearch results.
    - axes (2D np.ndarray): Array of axes with results plotted.
    """

    def single_evaluation_run(params, param_keys, **kwargs):
        if len(params) != len(param_keys):
            raise RuntimeError("'params' and 'param_keys' must have the same length.")
        params_dict = {key: param for key, param in zip(param_keys, params)}
        loss = evaluate_kernel(
            **params_dict,
            **kwargs,
        )
        return loss

    param_keys, all_param_values = zip(*kernel_search_dict.items())

    best_loss = np.inf
    param_grid = list(itertools.product(*all_param_values))

    if debug:
        losses = list()
        for params in tqdm(param_grid):
            losses.append(single_evaluation_run(params, param_keys, **kwargs))
    else:
        num_cpus = min(multiprocessing.cpu_count(), len(param_grid))
        losses = Parallel(n_jobs=num_cpus)(
            delayed(single_evaluation_run)(params, param_keys, **kwargs)
            for params in tqdm(param_grid)
        )

    best_indices = np.sort(np.where(losses == np.min(losses))[0])
    best_loss = losses[best_indices[0]]

    loss_array = np.zeros(list([len(values) for values in all_param_values]))
    for idx, values in enumerate(param_grid):
        grid_idx = list()
        for all_values, val in zip(all_param_values, values):
            grid_idx.append(np.where(all_values == val)[0][0])
        loss_array[*grid_idx] = losses[idx]

    best_index = get_best_index(param_grid, best_indices)
    best_values = param_grid[best_index]

    result_dict = {key: val for key, val in zip(param_keys, best_values)}

    result_dict["best_loss"] = best_loss
    result_dict["loss_array"] = loss_array

    inf_mask = np.isinf(loss_array)
    if len(loss_array[~inf_mask]) == 0:
        raise ValueError("No valid loss values found.")

    if log or plot_results:
        param_strs = [
            f"{key.capitalize().replace('_', ' ')}: {val:.4f}"
            for key, val in zip(param_keys, best_values)
        ]

    if log:
        max_loss = np.nanmax(loss_array[~inf_mask])
        param_str = "\n  ".join(param_strs)
        print(
            f"Best parameters:\n  {param_str}"
            f"\nBest loss: {best_loss:.4f} (max loss: {max_loss:.4f})"
        )

    if plot_results:
        if len(all_param_values) != 3:
            raise NotImplementedError(
                "Plotting results only implemented for 3 sets of parameters."
            )
        num_cols = min(5, len(all_param_values[-1]))
        num_rows = int(np.ceil(len(all_param_values[-1]) / num_cols))
        figsize = (num_cols * 1.4, num_rows * 1.55)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        title = f"Best parameters: {', '.join(param_strs)}"
        fig.suptitle(title, y=1.03, fontsize="small")

        best_i, best_j, best_k = np.unravel_index(best_index, loss_array.shape)

        labels = list()
        for param_key in param_keys:
            labels.append(
                param_key.capitalize()
                .replace("_", " ")
                .replace("weight", "wei.")
                .replace("filter ", "")
            )
        xlabel, ylabel, main_title = labels

        loss_array_finite = loss_array.copy()
        loss_array_finite[inf_mask] = loss_array[~inf_mask].max() * 1.5
        vmax = min(loss_array_finite.min() * 10, loss_array_finite.max())
        for k, sub_ax in enumerate(axes.ravel()):
            if k >= len(all_param_values[-1]):
                sub_ax.axis("off")
                continue

            sub_ax.imshow(loss_array_finite[:, ::-1, k].T, vmin=0, vmax=vmax)

            if k == best_k:
                i, j, s = best_i, best_j, 3
            else:
                i, j = np.unravel_index(
                    loss_array_finite[:, :, k].argmin(),
                    loss_array_finite[:, :, k].shape,
                )
                s = 0.5

            # mark best index
            j = loss_array.shape[1] - j - 1
            sub_ax.scatter(i, j, s=s, color="white", marker="*")

            min_val = loss_array_finite[..., k].min()
            title = f"{main_title}: {all_param_values[-1][k]:.2f}\n(min: {min_val:.4f})"
            sub_ax.set_title(title)
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])

            if k >= len(all_param_values[-1]) - num_cols:
                sub_ax.set_xlabel(xlabel, fontsize="x-small")
            if k % num_cols == 0:
                sub_ax.set_ylabel(ylabel, fontsize="x-small")

        return result_dict, axes

    else:
        return result_dict


def plot_kernel(
    sigma_in_steps=None,
    report_eval=True,
    eval_dict=dict(),
    plot_unsmoothed=True,
    sub_ax=None,
    **kwargs,
):
    """
    plot_kernel()

    Plot the kernel.

    Args:
    - sigma_in_steps (float, optional): Sigma for Gaussian filter in steps.
        Default is None.
    - report_eval (bool, optional): Whether to report the kernel evaluation.
        Default is False.
    - eval_dict (dict, optional): Dictionary of additional keyword argument for
        evaluating kernel. Default is dict().
    - plot_unsmoothed (bool, optional): Whether to plot the unsmoothed kernel as well,
        if applicable. Default is True.
    - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.

    Keyword args:
    - **kwargs: Keyword arguments passed to get_smoothed_summed_exp_vals() and
        evaluate_kernel().

    Returns:
    - sub_ax (plt.Axes): Subplot with kernel plotted.
    """

    if report_eval:
        evaluate_kernel(
            debug=True, sigma_in_steps=sigma_in_steps, **kwargs, **eval_dict
        )

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(7, 2))

    kernel_dict = get_complete_kernel_dict(**kwargs)

    plot_util.plot_summed_exp_kernel(
        sub_ax=sub_ax,
        sigma_in_steps=sigma_in_steps,
        target_root_dict=get_target_root_dict(),
        plot_unsmoothed=plot_unsmoothed,
        **kernel_dict,
    )

    return sub_ax


def plot_kernel_from_results_dict(results_dict, dt=0.03, sigma_in_steps=None, **kwargs):
    """
    plot_kernel_from_results_dict(results_dict)

    Plot the kernel from the results dictionary.

    Args:
    - results_dict (dict): Dictionary of gridsearch results.
    - dt (float, optional): Time step. Default is 0.03.
    - sigma_in_steps (float, optional): Sigma for Gaussian filter in steps.
        Default is None.

    Keyword args:
    - **kwargs: Keyword arguments passed to plotting passed to plot_kernel().

    Returns:
    - sub_ax (plt.Axes, optional): Subplot  with kernel plotted.
    """

    kernel_dict = {
        key.replace("best_", ""): val
        for key, val in results_dict.items()
        if "loss" not in key
    }

    sub_ax = plot_kernel(
        dt=dt,
        sigma_in_steps=sigma_in_steps,
        **kernel_dict,
        **kwargs,
    )

    return sub_ax


def get_kernel_search_dict(space="deltas"):
    """
    get_kernel_search_dict()

    Obtain a dictionary of values for the kernel hyperparameter search.

    Returns:
    - args_dict (dict): Dictionary of hyperparameter search values.
    """

    args_dict = {
        "pre_filter_tau_pos": (1.6, 2.5, 10),
        "pre_neg_delta": (0.005, 0.014, 9),
        "pre_neg_weight": (1.0007, 1.0016, 9),
        "post_filter_tau_pos": (0.9, 1.8, 10),
        "post_neg_delta": (0.005, 0.014, 9),
        "post_neg_weight": (1.0003, 1.0012, 9),
    }

    if space == "deltas":
        use_keys = ["pre_neg_delta", "post_neg_delta", "pre_filter_tau_pos"]
    elif space == "pre":
        use_keys = ["pre_filter_tau_pos", "pre_neg_delta", "pre_neg_weight"]
    elif space == "post":
        use_keys = ["post_filter_tau_pos", "post_neg_delta", "post_neg_weight"]
    else:
        raise ValueError(f"Invalid search space: {space}")

    kernel_search_dict = {key: np.linspace(*args_dict[key]) for key in use_keys}

    return kernel_search_dict


def get_min_neg_value(min_neg=None, side="pre"):
    """
    get_min_neg_value()

    Obtain the minimum negative value the kernel must reach.

    Args:
    - min_neg (float, optional): Minimum negative value. Default is None.
    - side (str, optional): Side of the kernel to optimize. Default is "pre".

    Returns:
    - min_neg (float): Minimum negative value.
    """

    if min_neg is None:
        if side == "pre":
            min_neg = 0.15
        elif side == "post":
            min_neg = 0.2
        else:
            raise ValueError(f"Invalid side: {side}")

    if min_neg < 0:
        raise ValueError(f"Minimum negative value must be non-negative: {min_neg}.")

    return min_neg


def main_kernel():
    """
    main_kernel()

    Run the kernel hyperparameter search.

    Returns:
    - full_dict (dict): Dictionary of gridsearch results and analysis parameters.
    - search_axes (2D np.ndarray): Array of axes with search results plotted.
    - kernel_ax1D (matplotlib.axes.Axes): Axes with kernel plotted.
    """

    args = get_kernel_args()
    kernel_search_dict = get_kernel_search_dict(args.space)

    if args.sigma is None:
        sigma_in_steps = None
    else:
        sigma_in_steps = ext_util.get_sigma_in_steps(
            sigma=float(args.sigma), dt=args.dt
        )

    kernel_kwargs = {
        "dt": args.dt,
        "sigma_in_steps": sigma_in_steps,
    }

    print(f"\nSearch\n{'-' * 6}")
    results_dict, search_axes = run_kernel_gridsearch(
        kernel_search_dict,
        log=True,
        plot_results=len(kernel_search_dict) == 3,
        debug=args.debug,
        **kernel_kwargs,
    )

    print(f"\nIdentified kernel\n{'-' * 17}")
    kernel_sub_ax = plot_kernel_from_results_dict(
        results_dict, report_eval=True, **kernel_kwargs
    )

    kernel_dict = params_util.get_default_BTSP_filter_param_dict(
        incl_BTSP_str=False, neg_delta=True
    )
    for key in kernel_search_dict.keys():
        if key in kernel_dict.keys():
            kernel_dict.pop(key)

    full_dict = {
        "kernel_search_dict": kernel_search_dict,
        "kernel_dict": kernel_dict,
        "kernel_kwargs": kernel_kwargs,
        "results_dict": results_dict,
    }

    if args.save:
        direc = get_save_directory(kernel=True)
        direc.mkdir(parents=True, exist_ok=True)
        date_time_str = gen_util.get_date_time_str()

        print(f"\nSaving results under '{direc}' with suffix '{date_time_str}'.")

        save_name = f"search_{args.space}_{date_time_str}"
        fig = search_axes.ravel()[0].figure
        for suffix in [".png", ".svg"]:
            save_path = Path(direc, f"{save_name}{suffix}")
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        save_name = f"kernel_{args.space}_{date_time_str}"
        fig = kernel_sub_ax.figure
        for suffix in [".png", ".svg"]:
            save_path = Path(direc, f"{save_name}{suffix}")
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        save_name = f"data_{args.space}_{date_time_str}"
        with open(Path(direc, f"{save_name}.pkl"), "wb") as f:
            pkl.dump(full_dict, f)

    return (
        full_dict,
        search_axes,
        kernel_sub_ax,
    )


def get_kernel_args():
    """
    get_kernel_args()

    Obtain arguments for the kernel hyperparameter search.

    Returns:
    - args (argparse.Namespace): Arguments for the hyperparameter search.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--space",
        default="deltas",
        choices=["deltas", "pre", "post"],
        help="Kernel search space.",
    )
    parser.add_argument("--dt", default=0.03, type=float, help="Update steps (s).")
    parser.add_argument("--sigma", default=None, help="Gaussian smoothing sigma (m).")

    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--plot_kernel", action="store_true", help="Plot kernel.")
    parser.add_argument("--save", action="store_true", help="Save results.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    plot_util.stylize_plots_for_notebook()
    results_dict, search_axes, kernel_sub_ax = main_kernel()
