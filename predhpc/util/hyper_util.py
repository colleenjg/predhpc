from pathlib import Path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from predhpc.util import params_util


def get_save_directory(save_directory=None):
    """
    get_save_directory()

    Obtain, and create if necessary, the save directory for hyperparameter search
    results.

    Args:
    - save_directory (str or Path): Directory to save results in. If None, a default
        directory is used (../results/hyperparameter_search). Default is None.

    Returns:
    - save_directory (Path): Directory to save results in.
    """

    if save_directory is None:
        save_directory = Path("..", "results", "hyperparameter_search")
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    return save_directory


def get_date_time_str():
    """
    get_date_time_str()

    Obtain the current date and time as a string.

    Returns:
    - date_time_str (str): Current date and time as a string.
    """

    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return date_time_str


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
    elif "time" in metric:
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


def get_metric_array(df, parameters=None, metric="num_BTSP_events", ascending=True):
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
    - im (matplotlib.image.AxesImage): Image from the plot.
    """

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

    if metric.startswith("num"):
        metric_array[metric_array == 0] = np.nan  # for plotting, only

    # start plotting
    if sub_ax is None:
        _, sub_ax = plt.subplots()

    vmin, vmax = get_metric_plot_vmin_vmax(metric)
    im = sub_ax.imshow(metric_array.T, cmap="viridis", vmin=vmin, vmax=vmax)

    sub_ax.set_title(title)
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

    return metric_array, im


def plot_metrics_by_parameters(df, parameters=None, save_directory=None, name_str=None):
    """
    plot_metrics_by_parameters(df)

    Plot metrics organized by parameters, with one metric in each subplot.

    Args:
    - df (pd.DataFrame): Dataframe in which hyperparameter search results are recorded.
    - parameters (list of str, optional): List of parameters to plot the metric by.
        If None, they are extracted from the dataframe
        (columns starting with 'config/'). Default is None.
    - save_directory (str or Path, optional): Directory to save results in. If None,
        the current working directory is used. Default is None.
    - name_str (str, optional): Name of the file in which to save the plot.
        Default is None.

    Returns:
    - axes (2D np.ndarray): Array of axes to plot on. There is one subplot per metric.
    """

    metrics = [
        col.replace("metric/", "") for col in df.columns if col.startswith("metric/")
    ]

    save_directory = get_save_directory(save_directory)

    if name_str is None:
        name_str = "metrics"

    ncols = int(np.ceil(np.sqrt(len(metrics))))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=[ncols * 6, nrows * 6], squeeze=False
    )
    for a, sub_ax in enumerate(axes.ravel()):
        if a < len(metrics):
            plot_metric_by_parameters(df, sub_ax, parameters, metric=metrics[a])
        else:
            sub_ax.axis("off")
    fig.savefig(Path(save_directory, f"{name_str}.png"), bbox_inches="tight", dpi=300)

    return axes


def run_hyperparameter_search(
    objective,
    search_space,
    dt=None,
    save_directory=None,
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
    - dt (float, optional): Time step for simulations. If None, a default time step is
        used. Default is None.
    - save_directory (str or Path, optional): Directory to save results in.
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

    dt = dt or params_util.DT

    ray.init(num_cpus=num_CPUs, log_to_driver=debug)

    # calculate number of runs (for grid search)
    num_combs = 1
    for hyperparam in search_space.values():
        num_combs *= len(hyperparam["grid_search"])
    print(
        f"\nRunning {num_combs} hyperparameter combinations ({num_repeats} repeats => "
        f"{num_repeats * num_combs} total runs) using {num_CPUs} CPUs.\n"
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

    save_directory = get_save_directory(save_directory)
    date_time_str = get_date_time_str()

    save_name = f"{save_name}_{date_time_str}"
    df.to_csv(Path(save_directory, f"{save_name}.csv"))

    parameters = list(search_space.keys())
    plot_metrics_by_parameters(
        df,
        parameters,
        save_directory=save_directory,
        name_str=save_name,
        **plot_kwargs,
    )

    return tuner
