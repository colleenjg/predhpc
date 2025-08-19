#!/usr/bin/env python3

from typing import Any, Sequence
import warnings

from matplotlib import pyplot as plt  # type: ignore[import]
import numpy as np
from tqdm import tqdm  # type: ignore[import]

from predhpc import agent, env, plot_fcts
from predhpc.neurons import (
    riab_neurons,
    learning_neurons,
    two_comp_neurons,
    object_neurons,
    value_neurons,
)
from predhpc.util import ext_util, gen_util, plot_util, params_util


class AdjustMaxTraj:
    def __init__(self, learner, finish_trajectory=True):
        self.learner = learner
        self.finish_trajectory = finish_trajectory
        self.original_max_num_traj = learner.max_num_traj

    def __enter__(self):
        if self.original_max_num_traj is not None:
            self.learner.max_num_traj = self.original_max_num_traj - 1

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_max_num_traj is not None:
            self.learner.max_num_traj = self.original_max_num_traj


class VNUpdater:

    def __init__(
        self,
        Agent,
        thresh_gradV=0.2,
        drift_vel_fact=3,
        drift_to_random_strength_ratio=0.2,
    ):
        self.Agent = Agent

        rewards = Agent.Environment.get_object_locations("reward")
        if len(rewards) != 1:
            raise ValueError(
                "VNUpdater requires exactly one reward object in the environment, "
                f"but found {len(rewards)}."
            )

        VN_params = params_util.get_VN_params(peak=rewards[0])
        VN = value_neurons.SimpleValueNeuron(Agent, params=VN_params)

        self.VN = VN
        self.thresh_gradV = thresh_gradV
        self.drift_vel_fact = drift_vel_fact
        self.drift_to_random_strength_ratio = drift_to_random_strength_ratio

    def get_update_kwargs(self):
        self.VN.update()
        drift_velocity = None
        if (
            self.Agent.target_position is not None
            and (self.Agent.target_position == self.VN.peak).all()
        ):
            gradV = self.VN.get_local_gradient(thresh_gradV=self.thresh_gradV)
            if gradV is not None:
                drift_velocity = self.drift_vel_fact * self.Agent.speed_mean * gradV
        update_kwargs = {
            "drift_velocity": drift_velocity,
            "drift_to_random_strength_ratio": self.drift_to_random_strength_ratio,
        }

        return update_kwargs


class Learner:
    """
    Learner

    Class for running a learning experiment with BTSP learning.
    """

    def __init__(
        self,
        Pyrs,
        reverse_linear=False,
        start_BTSP=None,
        stop_BTSP=None,
        BTSP_on=None,
        record_weights_at_BTSP=True,
        use_Hebbian=False,
        weight_recording_freq=100,
        max_num_target_reaches=None,
        max_num_traj=None,
    ):
        """
        Learner()

        Initialize a Learner object.

        Args:
        - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
            Pyr. neuron layer.
        - reverse_linear (bool, optional): If using a linear track, whether to reverse
            Agent diretion at each end. Default is False.
        - start_BTSP (int, optional): Step at which BTSP should start. Default is None.
        - stop_BTSP (int, optional): Step at which BTSP should stop. Default is None.
        - BTSP_on (int, optional): Trajectory number at which BTSP is enabled or
            triggered. 1 for first trajectory. Default is None.
        - record_weights_at_BTSP (bool, optional): Whether to record weights at BTSP
            events. Default is True.
        - use_Hebbian (bool, optional): Whether to use Hebbian learning. Default is
            False.
        - weight_recording_freq (int, optional): Frequency at which to record weights
            if Hebbian learning is active. Default is 100.
        - max_num_target_reaches (int, optional): Maximum number of target reaches.
            Default is None.
        - max_num_traj (int, optional): Maximum number of trajectories to complete.
            Default is None.
        """

        self.Pyrs = Pyrs

        self.record_weights_at_BTSP = record_weights_at_BTSP
        self.start_BTSP = start_BTSP
        self.stop_BTSP = stop_BTSP

        self.use_Hebbian = use_Hebbian
        self.weight_recording_freq = weight_recording_freq

        self.max_num_target_reaches = max_num_target_reaches
        self.max_num_traj = max_num_traj

        self.set_init_attributes(BTSP_on=BTSP_on, reverse_linear=reverse_linear)

    def get_agent_step(self):
        """
        self.get_agent_step()

        Get the current step of the agent.

        Returns:
        - agent_step (int): Current step of the agent.
        """

        agent_step = len(self.Agent.history["t"])

        return agent_step

    def set_init_attributes(self, BTSP_on=None, reverse_linear=False):
        """
        self.set_init_attributes()

        Set initial attributes for the Learner object.

        Args:
        - BTSP_on (int, optional): Trajectory number at which BTSP is enabled or
            triggered. 1 for first trajectory. Default is None.
        - reverse_linear (bool, optional): If using a linear track, whether to reverse
            Agent diretion at each end. Default is False.
        """

        self.Env, self.Agent, self.PCs, self.Objs = ext_util.extract_objects_from_Pyrs(
            self.Pyrs
        )

        # Agent info
        self.step = -1
        self.agent_start_step = self.get_agent_step()
        self.num_prev_traj_compl = self.Agent.get_num_completed_trajectories()
        self.num_prev_target_reaches = len(self.Agent.get_reached_target_df())
        self.traj_restarted = False
        self.early_stop_in_n = -1

        # Pyrs info
        self.two_compartment = isinstance(self.Pyrs, two_comp_neurons.TwoCompLayer)
        self.one_comp_internal = isinstance(self.Pyrs, learning_neurons.NMDALayer)

        # BTSP settings
        if self.two_compartment:
            self.Pyrs_for_weights = self.Pyrs.SomaCompartment
            self.BTSP_on = BTSP_on or 1
            self.BTSP_stopped = False
            self.Pyrs.set_BTSP_learn(soma=False, dend=False)
            self.Pyrs.set_learn(soma=self.use_Hebbian, dend=False, inhibit=False)
        else:
            self.Pyrs_for_weights = self.Pyrs
            if self.one_comp_internal:
                self.BTSP_on = BTSP_on or 1
            else:
                self.BTSP_on = BTSP_on or 3
            self.Pyrs.set_BTSP_learn()
            self.Pyrs.set_learn(self.use_Hebbian)

        # tracking BTSP
        self.BTSP_started = False
        self.BTSP_stopped = False
        self.num_BTSP_prev = len(self.Pyrs_for_weights.history["BTSP_events"])
        self.steps_BTSP_triggered = list()
        self.BTSP_neurons = list()

        # weight recording
        self.weights = [self.Pyrs_for_weights.inputs["PCs"]["w"].copy()]
        self.weight_steps = [self.agent_start_step]
        self.steps_triggered = [None]

        # reversal
        self.reverse_linear = reverse_linear
        if self.Env.D == 1:
            self.positions = [self.Agent.start_position, self.Agent.reset_position]
            self.check_pt = 1
        elif reverse_linear:
            raise ValueError("reverse_linear can only be used with a 1D environment.")

    def check_start_BTSP(self):
        """
        self.check_start_BTSP()

        Check whether BTSP should be started.
        """

        if not (self.two_compartment or self.one_comp_internal):
            return

        if self.start_BTSP is not None:
            return
        if len(self.Agent.trajectory_df) < self.BTSP_on + self.num_prev_traj_compl:
            return

        if self.two_compartment:
            self.Pyrs.set_BTSP_learn(soma=True, dend=False)
        else:
            self.Pyrs.set_BTSP_learn()

        self.start_BTSP = self.step

    def check_stop_BTSP(self, no_logs=False):
        """
        self.check_stop_BTSP()

        Check whether BTSP should be stopped.
        """

        if self.BTSP_stopped:
            return
        if self.stop_BTSP is None:
            return
        if self.step < self.stop_BTSP:
            return

        if self.two_compartment:
            self.Pyrs.set_BTSP_learn(soma=False, dend=False)
        else:
            self.Pyrs.set_BTSP_learn()

        self.BTSP_stopped = True

        if not no_logs:
            print(f"BTSP blocked from step {self.step + self.agent_start_step}.")

    def get_BTSP_targets(self):
        """
        self.get_BTSP_targets()

        Get targets for BTSP learning.
        """

        BTSP_targets = list()

        if self.two_compartment or self.one_comp_internal:
            return BTSP_targets

        if len(self.Agent.trajectory_df) != self.BTSP_on + self.num_prev_traj_compl:
            return BTSP_targets

        if self.BTSP_stopped:
            return BTSP_targets

        if self.start_BTSP is None:
            self.start_BTSP = self.step

        if self.traj_restarted and self.Pyrs.n > 1:  # BTSP near start position
            BTSP_targets = [self.Pyrs.n - 1]

        # check whether a target BTSP signal should be applied
        if self.Agent.reached_target:
            BTSP_targets = [0]

        return BTSP_targets

    def update_for_BTSP(self, no_logs=False):
        """
        self.update_for_BTSP()

        Update for BTSP learning.

        Returns:
        - BTSP_targets (list or None): List of target indices for BTSP learning or None.
        """
        # check for BTSP on previous update
        num_BTSP = len(self.Pyrs_for_weights.history["BTSP_events"])
        if num_BTSP > len(self.steps_BTSP_triggered) + self.num_BTSP_prev:
            self.steps_BTSP_triggered.append(self.get_agent_step() - 1)
            self.BTSP_neurons.append(self.Pyrs_for_weights.history["BTSP_targets"][-1])

        # check for BTSP on current update
        if self.two_compartment or self.one_comp_internal:
            self.check_start_BTSP()
            self.check_stop_BTSP(no_logs=no_logs)
            BTSP_targets = None
        else:
            BTSP_targets = self.get_BTSP_targets()

        return BTSP_targets

    def record_weights(self):
        """
        self.record_weights()

        Record weights if applicable.
        """

        if self.record_weights_at_BTSP and self.Pyrs_for_weights.BTSP_applied.any():
            step_triggered = (
                self.step - self.Pyrs_for_weights.history["num_steps_to_apply_BTSP"][-1]
            ) + self.agent_start_step

        elif self.use_Hebbian and not self.step % self.weight_recording_freq:
            step_triggered = None

        else:
            return

        step = self.agent_start_step + self.step
        if step not in self.weight_steps:
            self.weights.append(self.Pyrs_for_weights.inputs["PCs"]["w"].copy())
            self.weight_steps.append(step)
            self.steps_triggered.append(step_triggered)

    def check_reverse(self, final=False):
        """
        self.check_reverse()

        Check whether the Agent should reverse direction.

        Args:
        - final (bool, optional): Whether the check is for the final step. Default is
            False.
        """

        if not self.reverse_linear:
            return

        if self.Env.D != 1:
            raise ValueError("Reversing only applies to 1D environment.")

        if final and self.check_pt == 0:
            reverse = True
        else:
            reverse = self.Agent.check_if_position_reached(
                self.positions[self.check_pt]
            )

        if reverse:
            self.Agent.reverse(reset=True)
            self.check_pt = 1 - self.check_pt

    def check_for_early_stop(self):
        """
        self.check_for_early_stop()

        Check whether the learning should be stopped early.

        Returns:
        - (bool): Whether the learning should be stopped early.
        """

        if self.early_stop_in_n < 0:
            if self.max_num_target_reaches is not None:
                max_num_target_reaches = (
                    len(self.Agent.get_reached_target_df())
                    - self.num_prev_target_reaches
                )
                if max_num_target_reaches >= self.max_num_target_reaches:
                    self.early_stop_in_n = 20
            elif self.max_num_traj is not None:
                total_traj_compl = self.Agent.get_num_completed_trajectories()
                if total_traj_compl - self.num_prev_traj_compl >= self.max_num_traj:
                    self.early_stop_in_n = 20
        else:
            if self.early_stop_in_n == 0:
                return True
            else:
                self.early_stop_in_n -= 1

        return False

    def check_BTSP_enabled(self):
        """
        self.check_BTSP_enabled()

        Check whether BTSP was enabled during learning.

        Returns:
        - BTSP_enabled (bool): Whether BTSP was enabled.
        """

        BTSP_enabled = True
        if self.start_BTSP is None:
            BTSP_enabled = False
        elif self.stop_BTSP is None:
            BTSP_enabled = True
        elif self.start_BTSP >= self.stop_BTSP:
            BTSP_enabled = False

        return BTSP_enabled

    def log(self):
        """
        self.log()

        Log information about the learning process.
        """

        act_target_reaches = (
            len(self.Agent.get_reached_target_df()) - self.num_prev_target_reaches
        )
        if (
            self.max_num_target_reaches is None
            or act_target_reaches >= self.max_num_target_reaches
        ):
            print(f"Reached target {act_target_reaches} times.")
        else:
            print(
                f"Only reached target {act_target_reaches} times "
                f"(target: {self.max_num_target_reaches})."
            )

        self.Agent.log_trajectory_stats_to_date()
        self.Agent.log_trajectory_stats_to_date(log_as_time=False)

        steps_BTSP_triggered = np.asarray(self.steps_BTSP_triggered)

        if self.check_BTSP_enabled():
            if len(steps_BTSP_triggered) == 0:
                BTSP_stat_str = ""
            elif len(steps_BTSP_triggered) == 1:
                BTSP_stat_str = f": occurred at step {steps_BTSP_triggered[0]}"
            else:
                BTSP_stat_str = (
                    f": occurred between steps {steps_BTSP_triggered.min()} "
                    f"and {steps_BTSP_triggered.max()}, inclusively"
                )
            stop_BTSP = self.stop_BTSP or self.step
            event_str = "event" if len(steps_BTSP_triggered) == 1 else "events"
            neuron_str = ""
            if self.Pyrs_for_weights.n > 1 and len(steps_BTSP_triggered):
                n = len(np.unique(np.concatenate(self.BTSP_neurons)))
                neuron_str = f" in {n} neuron" if n == 1 else f" in {n} neurons"
            print(
                f"{len(steps_BTSP_triggered)} BTSP {event_str} triggered{neuron_str} "
                f"(allowed from steps {self.start_BTSP + self.agent_start_step} to "
                f"{stop_BTSP + self.agent_start_step}){BTSP_stat_str}."
            )
        else:
            print("BTSP not allowed.")

    def update(self, updater=dict(), no_logs=False):
        """
        self.update()

        Update the learning process.

        Args:
        - updater (object or dict, optional): Object or dictionary for updating
            agent position. Default is dict().
        - no_logs (bool, optional): Whether to disable logging. Default is False.

        Returns:
        - stop (bool): Whether the learning should be stopped.
        """

        self.step += 1

        if isinstance(updater, dict):
            update_kwargs = updater
        else:
            update_kwargs = updater.get_update_kwargs()

        self.Agent.update(**update_kwargs)
        if self.Objs is not None:
            self.Objs.update()
        self.PCs.update()

        BTSP_targets = self.update_for_BTSP(no_logs=no_logs)

        if self.two_compartment or self.one_comp_internal:
            self.Pyrs.update()
        else:
            self.Pyrs.update(BTSP_targets=BTSP_targets)

        self.record_weights()

        stop = self.check_for_early_stop()

        self.traj_restarted = self.Agent.reached_end
        if not stop:
            self.check_reverse()

        return stop

    def wrap_up(self, no_logs=False):
        """
        self.wrap_up()

        Wrap up the learning process.

        Args:
        - no_logs (bool, optional): Whether to disable logging. Default is False.
        """

        if not no_logs:
            self.log()

        if not self.check_BTSP_enabled() and len(self.steps_BTSP_triggered) > 0:
            raise RuntimeError(
                "BTSP events triggered even though BTSP was never enabled."
            )

        if self.early_stop_in_n != -1:
            self.early_stop_in_n = -1  # reset early stop counter

        return

    def get_most_BTSP_neurons(self):
        """
        self.get_most_BTSP_neurons()

        Get the neurons with the most BTSP events.

        Returns:
        - most_BTSP_neuron_idxs (1D array): Indices of the neurons with the most BTSP
            events.
        """

        idxs, counts = np.unique(np.concatenate(self.BTSP_neurons), return_counts=True)

        if len(idxs):
            most_BTSP_neuron_idxs = np.sort(idxs[counts == counts.max()])
        else:
            most_BTSP_neuron_idxs = np.arange(self.Pyrs_for_weights.n)

        num_BTSP = counts.max()

        return most_BTSP_neuron_idxs, num_BTSP

    def get_recorded_weights(self):
        """
        self.get_recorded_weights()

        Returns recorded weights.

        Returns:
        - recorded_weights (dict): Dictionary with keys "weights", "steps", "time", and
            "steps_triggered" in which input weights from place cells are recorded,
            along with the step/time at which they were recorded and step at which they
            the BTSP update behind the recorded weight update was triggered, if
            applicable. None if self.record_weights_at_BTSP and self.use_Hebbian are
            False.
        """

        recorded_weights = None
        if self.record_weights_at_BTSP or self.use_Hebbian:
            recorded_weights = ext_util.create_weights_dict(
                self.weights,
                self.weight_steps,
                t=self.Agent.history["t"],
                steps_triggered=self.steps_triggered,
            )

        return recorded_weights


def init_env_objects(
    env_params: dict[str, Any] | None = None,
    agent_params: dict[str, Any] | None = None,
    PC_params: dict[str, Any] | None = None,
    Pyr_params: dict[str, Any] | None = None,
    Obj_params: dict[str, Any] | None = None,
    environment="openfield",
    autosave: bool | None = None,
    plot: bool = True,
):
    """
    init_env_objects()

    Initialize objects for an environment, and obtain Pyrs.

    Args:
    - env_params (dict, optional): Parameters for the environment. Default is None.
    - agent_params (dict, optional): Parameters for the agent. Default is None.
    - PC_params (dict, optional): Parameters for the place cells. Default is
        None.
    - Pyr_params (dict, optional): Parameters for the Pyr. neurons. Default is None.
    - Obj_params (dict, optional): Parameters for the object neurons. Default is None.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.
    - plot (bool, optional): Whether to plot the environment and neurons. Default is
        True.

    Returns:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer.
    if plot and 2D environment:
    - fields_axes (2D np.ndarray): Array of subplots with place fields plotted, with
        shape (num_layers, num_samples).
    - aggreg_ax1D (1D np.ndarray): Array of subplots with environment and aggregated
        fields plotted, with shape (3,).
    if plot and 1D environment:
    - spatial_axes (2D np.ndarray): Array of subplots with 1D environment plotted, with
        shape (8, 1).
    """
    env_params = env_params or params_util.get_env_params(environment=environment)
    agent_params = agent_params or params_util.get_agent_params(environment=environment)

    if environment == "linear":
        Env = env.Environment(params=env_params)
        Ag = agent.ResetableAgent(Env, params=agent_params)
    elif environment == "tmaze":
        Env = env.TEnv(params=env_params)
        Ag = agent.TAgent(Env, params=agent_params)
    elif environment in ["openfield", "openfield_corridor"]:
        Env = env.OpenField(params=env_params)
        Ag = agent.OpenFieldAgent(Env, params=agent_params)
    else:
        raise ValueError(f"Invalid environment: {environment}")

    PC_params = PC_params or params_util.get_PC_params(environment=environment)
    PCs = riab_neurons.PlaceCells(Ag, params=PC_params)

    # infer whether two-compartment model will be used or not
    if Pyr_params is None or any(key.startswith("soma_") for key in Pyr_params.keys()):
        two_compartment = True
    else:
        two_compartment = False

    if two_compartment:
        Obj_params = Obj_params or params_util.get_Obj_params(environment=environment)
        if environment in ["linear", "tmaze"]:
            Obj_type = object_neurons.ObjectCells
        else:
            fixed = any([key.startswith("num_") for key in Obj_params.keys()])
            Obj_type = (
                object_neurons.FixedObjectCells
                if fixed
                else object_neurons.ObjectInstanceCells
            )
        Objs = Obj_type(Ag, params=Obj_params)
    else:
        if Obj_params is not None:
            warnings.warn("Obj_params will be ignored if two_compartment is False.")
        Objs = None

    if Pyr_params is None:
        n_kwargs = {"n": Objs.n} if two_compartment else dict()
        Pyr_params = params_util.get_Pyr_params(
            environment=environment,
            BTSP=True,
            NMDA=two_compartment,
            two_compartment=two_compartment,
            **n_kwargs,
        )

    if two_compartment:
        Pyr_params["soma_input_layers"] = [PCs]  # type: ignore[assignment]
        if Pyr_params["n"] is None:
            Pyr_params["n"] = Objs.n
        elif Pyr_params["n"] != Objs.n:
            raise ValueError(
                f"If provided, Pyr_params['n'] ({Pyr_params['n']}) must be equal to "
                f"Objs.n ({Objs.n})."
            )
    else:
        Pyr_params["input_layers"] = [PCs]  # type: ignore[assignment]

    if two_compartment:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No input layers")
            Pyrs = two_comp_neurons.TwoCompLayer(Ag, params=Pyr_params)

        Obj_to_Pyr_w = gen_util.get_weights(
            Objs.n,
            Pyrs.n,
            loc=Pyr_params["dend_w_init_loc"],
            scale=Pyr_params["dend_w_init_scale"],
        )
        Pyrs.DendriteCompartment.add_input(Objs, w=Obj_to_Pyr_w)
        Pyrs.set_BTSP_learn(soma=True, dend=False)
    else:
        if "NMDA_activation_threshold" in Pyr_params.keys():
            Pyrs = learning_neurons.NMDALayer(Ag, params=Pyr_params)
        else:
            Pyrs = learning_neurons.BTSPLayer(Ag, params=Pyr_params)
        Pyrs.set_BTSP_learn()

    if plot:
        if environment in ["tmaze", "openfield", "openfield_corridor"]:
            fields_axes, aggreg_ax1D = plot_fcts.plot_2D_initial_conditions(
                Pyrs, autosave=autosave
            )
            return Pyrs, fields_axes, aggreg_ax1D
        else:
            spatial_axes = plot_fcts.plot_1D_initial_conditions(Pyrs)
            return Pyrs, spatial_axes

    else:
        return Pyrs


def finish_learn_trajectory(learner, updater=dict(), no_logs=False):
    """
    finish_learn_trajectory(learner)

    Finish the current trajectory for a learner.

    Args:
    - learner (Learner): Learner object.
    - updater (object or dict, optional): Object or dictionary for updating
        agent position. Default is dict().
    - no_logs (bool, optional): Whether to disable logging. Default is False.
    """

    if learner.Agent.reached_end:
        return

    if not no_logs:
        print("Finishing last trajectory.")

    def generator():
        while True:
            yield

    break_next = False
    for _ in tqdm(generator(), disable=no_logs):
        learner.update(updater=updater, no_logs=no_logs)
        if break_next:
            break
        if learner.Agent.reached_end:
            break_next = True


def run_learner(
    learner,
    updater=dict(),
    max_num_steps=10000,
    finish_trajectory=False,
    no_logs=False,
):
    """
    run_learner()

    Run a learning experiment with a learner.

    Args:
    - learner (Learner): Learner object.
    - updater (object or dict, optional): Object or dictionary for updating
        agent position. Default is dict().
    - max_num_steps (int, optional): Maximum number of steps to run. Will constrain
        other stopping conditions (number of target reaches or trajectories). Pass None
        to avoid constraining these by number of steps, and learning will only stop
        when one of those conditions are reached, if provided. Default is 10000.
    - finish_trajectory (bool, optional): Whether to finish the last trajectory.
        Default is False.
    - no_logs (bool, optional): Whether to disable logging. Default is False.
    """

    with AdjustMaxTraj(learner, finish_trajectory=finish_trajectory):
        if max_num_steps is None:

            def infinite_generator():
                while True:
                    yield

            generator = infinite_generator()
        else:
            generator = range(max_num_steps)

        for _ in tqdm(generator, disable=no_logs):
            stop = learner.update(updater=updater, no_logs=no_logs)

            if stop:
                break

    if finish_trajectory:
        finish_learn_trajectory(learner, no_logs=no_logs)

    learner.wrap_up(no_logs=no_logs)

    return


def learn(
    Pyrs_or_learner,
    max_num_target_reaches=None,
    max_num_traj=None,
    max_num_steps=10000,
    finish_trajectory=False,
    record_weights_at_BTSP=True,
    weight_recording_freq=100,
    use_Hebbian=False,
    BTSP_on=None,
    num_end_without_BTSP=0,
    reverse_linear=False,
    updater=dict(),
    no_logs=False,
):
    """
    learn(Pyrs_or_learner)

    Run a learning experiment with BTSP learning.

    Args:
    - Pyrs_or_learner (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer or Learner):
        Pyr. neuron layer or learner.
    - max_num_target_reaches (int or None, optional): Maximum number of target reaches.
        Default is None.
    - max_num_traj (int or None, optional): Maximum number of trajectories to complete.
        Default is None.
    - max_num_steps (int or None, optional): Maximum number of steps to run. Will constrain
        other stopping conditions (number of target reaches or trajectories). Pass None
        to avoid constraining these by number of steps, and learning will only stop
        when one of those conditions are reached, if provided. Default is 10000.
    - finish_trajectory (bool, optional): Whether to finish the last trajectory.
        Default is False.
    - record_weights_at_BTSP (bool, optional): Whether to record weights at BTSP events.
        Default is True.
    - weight_recording_freq (int, optional): Frequency at which to record weights if
        Hebbian learning is active. Default is 100.
    - use_Hebbian (bool, optional): Whether to use Hebbian learning. Default is False.
    - BTSP_on (int, optional): Trajectory number at which BTSP is enabled or
        triggered. 1 for first trajectory. Default is None.
    - num_end_without_BTSP (int, optional): Number of final steps to run without BTSP
        learning. Default is 0.
    - reverse_linear (bool, optional): If using a linear track, whether to reverse
        Agent diretion at each end. Default is False.
    - updater (object or dict, optional): Object or dictionary for updating
        agent position. Default is dict().
    - no_logs (bool, optional): Whether to disable logging. Default is False.

    Returns:
    - learner (Learner): Learner object.
    """

    stop_BTSP = None
    if num_end_without_BTSP:
        stop_BTSP = max(0, max_num_steps - num_end_without_BTSP)

    if isinstance(Pyrs_or_learner, Learner):
        learner = Pyrs_or_learner
        learner.max_num_target_reaches = max_num_target_reaches
        learner.max_num_traj = max_num_traj
    else:
        learner = Learner(
            Pyrs_or_learner,
            reverse_linear=reverse_linear,
            stop_BTSP=stop_BTSP,
            BTSP_on=BTSP_on,
            record_weights_at_BTSP=record_weights_at_BTSP,
            use_Hebbian=use_Hebbian,
            weight_recording_freq=weight_recording_freq,
            max_num_target_reaches=max_num_target_reaches,
            max_num_traj=max_num_traj,
        )

    run_learner(
        learner,
        updater=updater,
        max_num_steps=max_num_steps,
        finish_trajectory=finish_trajectory,
        no_logs=no_logs,
    )

    return learner


### 2D (OPENFIELD) FUNCTIONS ###


def learn_openfield_BTSP(
    Pyrs_or_learner: (
        learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer | Learner | None
    ) = None,
    max_num_steps: int | None = 10000,
    finish_trajectory: bool = False,
    record_weights_at_BTSP: bool = True,
    weight_recording_freq: int = 100,
    use_Hebbian: bool = False,
    num_end_without_BTSP: int = 0,
    corridor: bool = False,
    updater: dict[str, Any] | None = None,
    teleportation_enabled: bool | None = None,
    no_logs: bool = False,
    autosave: bool | None = None,
    **init_kwargs,
):
    """
    learn_openfield_BTSP()

    Run an openfield learning experiment with BTSP learning.

    Args:
    - Pyrs_or_learner (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer or Learner):
        Pyr. neuron layer or learner.
    - max_num_steps (int or None, optional): Maximum number of steps to run. Will
        constrain other stopping conditions (number of target reaches or trajectories).
        Pass None to avoid constraining these by number of steps, and learning will
        only stop when one of those conditions are reached, if provided.
        Default is 10000.
    - finish_trajectory (bool, optional): Whether to finish the last trajectory.
        Default is False.
    - record_weights_at_BTSP (bool, optional): Whether to record weights at BTSP events.
        Default is True.
    - weight_recording_freq (int, optional): Frequency at which to record weights if
        Hebbian learning is active. Default is 100.
    - use_Hebbian (bool, optional): Whether to use Hebbian learning. Default is False.
    - num_end_without_BTSP (int, optional): Number of final steps to run without BTSP
        learning. Default is 0.
        is True.
    - corridor (bool, optional): Whether to use the openfield corridor environment.
        Default is False.
    - updater (object or dict, optional): Object or dictionary for updating
        agent position. Default is None.
    - teleportation_enabled (bool, optional): Whether teleportation should be enabled.
        Default is None.
    - no_logs (bool, optional): Whether to disable logging. Default is False.
    - autosave (bool, optional): Whether to autosave. Default is None.

    Keyword Args:
    - **init_kwargs: Keyword arguments for init_env_objects().

    Returns:
    - learner (Learner): Learner object.
    """

    if Pyrs_or_learner is None:
        environment = "openfield_corridor" if corridor else "openfield"
        Pyrs_or_learner = init_env_objects(
            environment=environment,
            autosave=autosave,
            plot=False,
            **init_kwargs,
        )

    if isinstance(Pyrs_or_learner, Learner):
        Pyrs = Pyrs_or_learner.Pyrs
    else:
        Pyrs = Pyrs_or_learner
    if not isinstance(Pyrs.Agent.Environment, env.OpenField):
        raise ValueError("Pyrs must be an openfield environment.")

    if updater is None:
        if corridor:
            updater = VNUpdater(Pyrs.Agent)
        else:
            updater = {
                "speed_fact": 3,
                "drift_to_random_strength_ratio": 1,
            }

    if corridor:
        Pyrs.Agent.pos = np.asarray([0.2, 0.8])  # start position
        Pyrs.head_direction = np.asarray([-1, 0])

    if teleportation_enabled is not None:
        Pyrs.Agent.allow_teleportation(teleportation_enabled)

    learner = learn(
        Pyrs_or_learner,
        BTSP_on=0,
        max_num_steps=max_num_steps,
        finish_trajectory=finish_trajectory,
        record_weights_at_BTSP=record_weights_at_BTSP,
        weight_recording_freq=weight_recording_freq,
        use_Hebbian=use_Hebbian,
        num_end_without_BTSP=num_end_without_BTSP,
        updater=updater,
        no_logs=no_logs,
    )

    return learner


### 2D (T-MAZE) FUNCTIONS ###


def plot_T_maze(
    Pyrs_or_Objs: learning_neurons.BTSPLayer | object_neurons.ObjectCells,
    PCs: riab_neurons.PlaceCells | None = None,
    method: str = "groundtruth",
    autosave: bool | None = None,
):
    """
    plot_T_maze(Pyrs_or_Objs)

    Plot the T-maze environment:
        (1) Agent trajectories,
        (2) Place cell locations, and
        (3) Pyr. or Obj. overlayed rate maps.

    Args:
    - Pyrs_or_Objs (learning_neurons.BTSPLayer or object_neurons.ObjectCells):
        Pyrs layer.
    - PCs (riab_neurons.PlaceCells): Place cells. If not provided, will be extracted
        from Pyrs_or_Objs if it is a BTSPLayer. Default is None.
    - method (str, optional): Method to use for plotting the Pyr. rate map. Default
        is "groundtruth".
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - axes (2D np.ndarray): Array of subplots with T maze information plotted, with
        shape (3, 1). See description for details.
    """

    if isinstance(Pyrs_or_Objs, learning_neurons.BTSPLayer):
        _, Ag, extracted_PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs_or_Objs)
        PCs = PCs or extracted_PCs
    else:
        Ag = Pyrs_or_Objs.Agent
        if PCs is None:
            raise ValueError("PCs must be provided if Pyrs_or_Objs is not a BTSPLayer.")

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), squeeze=False)
    ax1D = np.asarray(axes).ravel()

    # Plot trajectories on T-maze
    Ag.plot_trajectories(scale_cmap_per=False, s_2D=5, alpha=0.3, sub_ax=ax1D[0])
    ax1D[0].set_title("Trajectories")

    # Plot place cell locations on T-maze
    plot_fcts.plot_overlayed_rate_maps(
        PCs, sub_ax=ax1D[1], method="max", colorbar=False
    )
    PCs.plot_place_cell_locations(sub_ax=ax1D[1])
    ax1D[1].scatter(
        *Ag.target_position,
        marker=".",
        color="blue",
        s=18,
        zorder=5,
    )
    ax1D[1].set_title("Place cell rate maps")

    # Plot Pyr. rate map on T-maze
    if isinstance(Pyrs_or_Objs, learning_neurons.BTSPLayer):
        Pyrs_or_Objs.plot_rate_map(ax=ax1D[2], method=method)
        title = f"{Pyrs_or_Objs.name.replace('_', ' ')} rate map"  # type: ignore[attr-defined]
    else:
        plot_fcts.plot_overlayed_rate_maps(
            Pyrs_or_Objs,
            sub_ax=ax1D[2],
            method="max",
            colorbar=False,
            plot_env=True,
        )
        title = "Obj. rate map"

    ax1D[2].scatter(
        *Ag.target_position,
        marker=".",
        color="blue",
        s=18,
        zorder=5,
    )
    ax1D[2].set_title(title)

    plot_util.save_figure(fig, "T_maze", save=autosave)

    return axes


def learn_T_maze_BTSP(
    Pyrs_or_learner: (
        learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer | Learner | None
    ) = None,
    max_num_target_reaches: int = 200,
    max_num_traj: int | None = None,
    max_num_steps: int | None = 10000,
    finish_trajectory: bool = True,
    record_weights_at_BTSP: bool = True,
    weight_recording_freq: int = 100,
    use_Hebbian: bool = False,
    BTSP_on: int | None = None,
    updater: dict[str, Any] | None = None,
    no_logs: bool = False,
    plot: bool = True,
    autosave: bool | None = None,
    **init_kwargs,
):
    """
    learn_T_maze_BTSP()

    Run a T-maze learning experiment with BTSP learning.

    Args:
    - Pyrs_or_learner (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer or Learner):
        Pyr. neuron layer or learner.
    - max_num_target_reaches (int or None, optional): Maximum number of target reaches.
        Default is 200.
    - max_num_traj (int or None, optional): Maximum number of trajectories to complete.
        Default is None.
    - max_num_steps (int or None, optional): Maximum number of steps to run. Will constrain
        other stopping conditions (number of target reaches or trajectories). Pass None
        to avoid constraining these by number of steps, and learning will only stop
        when one of those conditions are reached, if provided. Default is 10000.
    - finish_trajectory (bool, optional): Whether to finish the last trajectory.
        Default is True.
    - record_weights_at_BTSP (bool, optional): Whether to record weights at BTSP events.
        Default is True.
    - weight_recording_freq (int, optional): Frequency at which to record weights if
        Hebbian learning is active. Default is 100.
    - use_Hebbian (bool, optional): Whether to use Hebbian learning. Default is False.
    - BTSP_on (int, optional): Trajectory number at which BTSP is enabled or
        triggered. 1 for first trajectory. Default is None.
    - updater (object or dict, optional): Object or dictionary for updating
        agent position. Default is None.
    - no_logs (bool, optional): Whether to disable logging. Default is False.
    - plot (bool, optional): Whether to plot the environment and neurons. Default is
        True.
    - autosave (bool, optional): Whether to autosave. Default is None.

    Keyword Args:
    - **init_kwargs: Keyword arguments for init_env_objects().

    Returns:
    - learner (Learner): Learner object.
    if plot:
    - spatial_axes (2D np.ndarray): Array of subplots with T maze information plotted,
        with shape (3, 1). See run_manager.plot_T_maze() for details.
    - rate_maps_axes (2D np.ndarray): Array of subplots with rate maps across learning
        plotted, with shape (3, 1).
        See learning_neurons.BTSPLayer.plot_rate_maps_across_learning() for details.
    - BTSP_ax1D (1D np.ndarray): Subplots with BTSP events and proximity to target
        plotted. See plot_fcts.plot_time_series_with_BTSP_events() for details.
    """

    if Pyrs_or_learner is None:
        Pyrs_or_learner = init_env_objects(
            environment="tmaze",
            autosave=autosave,
            plot=False,
            **init_kwargs,
        )

    else:
        if isinstance(Pyrs_or_learner, Learner):
            Pyrs = Pyrs_or_learner.Pyrs
        else:
            Pyrs = Pyrs_or_learner
        if not isinstance(Pyrs.Agent.Environment, env.TEnv):
            raise ValueError("Pyrs must be a T-maze environment.")

    if updater is None:
        updater = {
            "speed_fact": 3,
            "drift_to_random_strength_ratio": 1,
        }

    learner = learn(
        Pyrs_or_learner,
        max_num_target_reaches=max_num_target_reaches,
        max_num_traj=max_num_traj,
        max_num_steps=max_num_steps,
        finish_trajectory=finish_trajectory,
        record_weights_at_BTSP=record_weights_at_BTSP,
        weight_recording_freq=weight_recording_freq,
        use_Hebbian=use_Hebbian,
        BTSP_on=BTSP_on,
        updater=updater,
        no_logs=no_logs,
    )

    if plot:
        if learner.Objs is None:
            spatial_axes = plot_T_maze(learner.Pyrs, autosave=autosave, method="groundtruth")  # type: ignore[arg-type]
        else:
            spatial_axes = plot_T_maze(learner.Objs, learner.PCs, autosave=autosave, method="history")  # type: ignore[arg-type]

        rate_maps_axes = learner.Pyrs.plot_rate_maps_across_learning()  # type: ignore[attr-defined]

        BTSP_ax1D = plot_fcts.plot_time_series_with_BTSP_events(learner.Pyrs)  # type: ignore[arg-type]

        return learner, spatial_axes, rate_maps_axes, BTSP_ax1D

    else:
        return learner


### 1D (LINEAR TRACK) FUNCTIONS ###


def learn_1D_BTSP(
    Pyrs_or_learner: (
        learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer | Learner | None
    ) = None,
    max_num_target_reaches: int = 10,
    max_num_traj: int | None = None,
    max_num_steps: int | None = 5000,
    finish_trajectory: bool = True,
    record_weights_at_BTSP: bool = True,
    weight_recording_freq: int = 100,
    use_Hebbian: bool = False,
    BTSP_on: int | None = None,
    reverse: bool = False,
    updater: dict[str, Any] = dict(),
    no_logs: bool = False,
    plot: bool = True,
    autosave: bool | None = None,
    **init_kwargs,
):
    """
    learn_1D_BTSP()

    Run a 1D learning experiment with BTSP learning. Plot spatial and time information.

    Args:
    - Pyrs_or_learner (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer, Learner, optional):
        Pyr. neurons. If None, will be initialized. Default is None.
    - max_num_target_reaches (int or None, optional): Maximum number of target reaches.
        Default is 10.
    - max_num_traj (int or None, optional): Maximum number of trajectories to complete.
        Default is None.
    - max_num_steps (int or None, optional): Maximum number of steps to run. Will constrain
        other stopping conditions (number of target reaches or trajectories). Pass None
        to avoid constraining these by number of steps, and learning will only stop
        when one of those conditions are reached, if provided. Default is 5000.
    - finish_trajectory (bool, optional): Whether to finish the last trajectory.
        Default is True.
    - record_weights_at_BTSP (bool, optional): Whether to record weights at BTSP events.
        Default is True.
    - weight_recording_freq (int, optional): Frequency at which to record weights.
        Default is 100.
    - use_Hebbian (bool, optional): Whether to use Hebbian learning.
        Default is False.
    - BTSP_on (int, optional): Trajectory number at which BTSP is enabled or
        triggered. 1 for first trajectory. Default is None.
    - reverse (bool, optional): Whether to reverse Agent diretion at each end.
        Default is False.
    - updater (object or dict, optional): Object or dictionary for updating
        agent position. Default is dict().
    - no_logs (bool, optional): Whether to disable logging. Default is False.
    - plot (bool, optional): Whether to plot the environment and neurons. Default is
        True.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Keyword Args:
    - **init_kwargs: Keyword arguments for init_env_objects().

    Returns:
    - learner (Learner): Learner object.
    if plot:
    - spatial_axes (2D np.ndarray): Array of subplots with 1D environment experiment
        info plotted, with shape (8, 1). See run_manager.plot_1D_spatial_info() for
        details.
    - time_axes (2D np.ndarray): Array of subplots with 1D time info plotted, with
        shape (3, 1). See run_manager.plot_1D_time_info() for details.
    """

    if Pyrs_or_learner is None:
        Pyrs_or_learner = init_env_objects(
            environment="linear",
            autosave=autosave,
            plot=False,
            **init_kwargs,
        )
    else:
        if isinstance(Pyrs_or_learner, Learner):
            Pyrs = Pyrs_or_learner.Pyrs
        else:
            Pyrs = Pyrs_or_learner

        if Pyrs.Agent.Environment.D != 1:
            raise ValueError("Pyrs must be a 1D environment.")

    learner = learn(
        Pyrs_or_learner,
        max_num_target_reaches=max_num_target_reaches,
        max_num_traj=max_num_traj,
        max_num_steps=max_num_steps,
        finish_trajectory=finish_trajectory,
        record_weights_at_BTSP=record_weights_at_BTSP,
        weight_recording_freq=weight_recording_freq,
        use_Hebbian=use_Hebbian,
        BTSP_on=BTSP_on,
        reverse_linear=reverse,
        updater=updater,
        no_logs=no_logs,
    )

    if plot:
        recorded_weights = learner.get_recorded_weights()
        weights = recorded_weights["weights"] if recorded_weights is not None else None
        spatial_axes = plot_fcts.plot_1D_spatial_info(
            learner.Pyrs, weights, autosave=autosave
        )

        time_axes = plot_fcts.plot_1D_time_info(learner.Pyrs, autosave=autosave)

        return learner, spatial_axes, time_axes

    else:
        return learner


if __name__ == "__main__":
    learner, spatial_axes, time_axes = learn_1D_BTSP(plot=True)

    breakpoint()
