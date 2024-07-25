from pathlib import Path
from typing import TYPE_CHECKING
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary as tsummary

if TYPE_CHECKING:
    from predhpc.neurons import trainable_neurons


class TorchNeuronModel(torch.nn.Module):
    """
    TorchNeuronModel()

    Torch neuron model that updates weights of a ratinabox neuron layer.
    """

    def __init__(
        self,
        seq_neuron_layers: list["trainable_neurons.TorchLayer"],
        device: str = "cpu",
        lr: float = 1e-4,
        RMSprop: bool = False,
    ):
        """
        TorchNeuronModel()

        Initialise a PyTorch model with a sequence of TorchLayer objects.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.
        - criterion (torch.nn.MSELoss): Mean squared error loss.
        - optimizer (torch.optim.Adam or torch.optim.RMSprop): Optimizer.
        - device (str): Device to run model on.
        - seq_layers (list[torch.nn.Module]): List of PyTorch layers.
        - seq_neuron_layers (list[trainable_neurons.TorchLayer]):
            List of torch neuron layers. Must feed into each other sequentially.

        Args:
        - seq_neuron_layers (list[neurons.TorchLayer]): List of TorchLayer objects.
        - device (str, optional): Device to run model on. Default is "cpu".
        - lr (float, optional): Learning rate. Default is 1e-4.
        - RMSprop (bool, optional): If True, use RMSprop optimizer instead of Adam.
            Default is False.
        """

        super().__init__()

        self.seq_neuron_layers = (
            seq_neuron_layers  # should feed into each other sequentially
        )

        self.seq_layers = list()
        for n, neuron_layer in enumerate(self.seq_neuron_layers):
            name = f"layer{n + 1}"
            setattr(self, name, neuron_layer.layer)
            self.seq_layers.append(getattr(self, name))

        self.criterion = torch.nn.MSELoss()

        self.set_device(device)

        optim_type = torch.optim.RMSprop if RMSprop else torch.optim.Adam
        self.optimizer = optim_type(self.parameters(), lr=lr)

        self.Agent = self.seq_neuron_layers[0].Agent

    def set_device(self, device: str = "cpu"):
        """
        self.set_device()

        Set the device to run the model on.

        Attributes:
        - device (str): Device to run model on.

        Args:
        - device (str, optional): Device to run model on. Default is "cpu".
        """

        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        """
        self.forward()

        Run a forward pass through the model.

        Args:
        - x (3D torch.Tensor): Input tensor with shape
            (time_steps, input_size).

        Returns:
        - x (torch.Tensor): Output tensor with shape
            (time_steps, output_size).
        """

        for layer in self.seq_layers:
            x = layer(x)
        return x

    def get_X(self, n):
        """
        self.get_X(n)

        Get the input tensor for the last n time steps.

        Args:
        - n (int): Number of time steps (most recent) to get.

        Raises:
        - ValueError: If fewer than n steps recorded.

        Returns:
        - X (torch.Tensor): Input tensor with shape
            (n time steps, input layer size).
        """

        if len(self.seq_neuron_layers[0].history["firingrate"]) < n:
            raise ValueError(f"Fewer than {n} steps recorded.")

        # shape (time, number of input neurons)
        X = torch.Tensor(
            np.concatenate(
                [
                    input_layer["layer"].history["firingrate"][-n:]
                    for input_layer in self.seq_neuron_layers[0].inputs.values()
                ],
                axis=1,
            )
        )

        return X

    def get_y(self, n: int) -> torch.Tensor:
        """
        self.get_y(n)

        Get the output tensor for the last n time steps.

        Args:
        - n (int): Number of time steps (most recent) to get.

        Raises:
        - ValueError: If fewer than n steps recorded.

        Returns:
        - y (torch.Tensor): Output tensor with shape
            (n time steps, output layer size).
        """

        if len(self.seq_neuron_layers[0].history["firingrate"]) < n:
            raise ValueError(f"Fewer than {n} steps recorded.")

        y = torch.Tensor(self.seq_neuron_layers[0].Agent.history["pos"][-n:])

        return y

    def run_train(self, n: int):
        """
        self.run_train(n)

        Run a training step, computing the loss and updating the weights.

        Args:
        - n (int): Number of time steps (most recent) to train on.

        Returns:
        - (float): Loss value.
        """
        self.train()

        pred = self(self.get_X(n).to(self.device))

        loss = self.criterion(pred, self.get_y(n).to(self.device))

        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()

        # step optimizer
        self.optimizer.step()
        self.zero_grad()

        for neuron_layer in self.seq_neuron_layers:
            neuron_layer.update_weights()

        return loss.item()

    def run_eval(self, n: int):
        """
        self.run_eval(n)

        Evaluate the model, computing the loss.

        Args:
        - n (int): Number of time steps (most recent) to use to evaluate model.

        Returns:
        - (float): Loss value.
        """

        self.eval()

        with torch.no_grad():
            pred = self(self.get_X(n).to(self.device))
            loss = self.criterion(pred, self.get_y(n).to(self.device))

        return loss.item()


class PredHPC(torch.nn.Module):
    """
    PredHPC()

    Torch predictive HPC model.
    """

    def __init__(
        self,
        input_size: int = 300,
        n_DG_CA3: int = 100,
        n_CA1: int = 100,
        pred_size: int = 2,
        summary: bool = True,
    ):
        """
        PredHPC()

        Initialise a predictive HPC model.

        Attributes:
        - CA1_dend_decoder (torch.nn.Linear): CA1 dendrite to predictive output layer.
        - DG_CA3_to_CA1_soma (torch.nn.Linear): DG/CA3 to CA1 soma layer.
        - input_size (int): Number of inputs to the network.
        - input_to_CA1_dend (torch.nn.Linear): Input to CA1 dendrite layer.
        - input_to_DG_CA3 (torch.nn.Linear): Input to DG/CA3 layer.
        - n_CA1 (int): Size of the CA1 layer.
        - n_DG_CA3 (int): Size of the DG/CA3 layer.
        - pred_size (int): Size of the predictive output layer.

        Args:
        - input_size (int, optional): Number of inputs to the network. Default is 300.
        - n_DG_CA3 (int, optional): Size of the DG/CA3 layer. Default is 100.
        - n_CA1 (int, optional): Size of the CA1 layer. Default is 100.
        - pred_size (int, optional): Size of the predictive output layer. Default is 2.
        - summary (bool, optional): If True, a summary of the model is printed to the
            console. Default is True.
        """

        super().__init__()

        self.input_size = input_size
        self.n_DG_CA3 = n_DG_CA3
        self.n_CA1 = n_CA1
        self.pred_size = pred_size

        # long path
        self.input_to_DG_CA3 = torch.nn.Linear(self.input_size, self.n_DG_CA3)
        self.DG_CA3_to_CA1_soma = torch.nn.Linear(self.n_DG_CA3, self.n_CA1)

        # convergence
        # self.CA1_convergence

        # direct path
        self.input_to_CA1_dend = torch.nn.Linear(self.input_size, self.n_CA1)
        self.CA1_dend_decoder = torch.nn.Linear(self.n_CA1, self.pred_size)

        if summary:
            tsummary(self, (input_size,))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        self.forward(X)

        Run a forward pass through the model.

        Args:
        - X (2D Torch tensor): Input activations with shape (batch_size, input_size).

        Returns:
        - predictions (2D Torch tensor): Model predictions with shape
            (batch_size, pred_size).
        """

        ## long path
        # self.DG_CA3 = F.relu(self.input_to_DG_CA3(x))
        # self.CA1_soma = F.relu(self.DG_CA3_to_CA1_soma(self.DG_CA3)) # dead-end for now

        # direct path
        self.CA1_dend = F.relu(self.input_to_CA1_dend(X))

        # convergence
        self.CA1 = self.CA1_dend

        # prediction
        predictions = F.relu(self.CA1_dend_decoder(self.CA1))

        return predictions


def save_model(
    model: torch.nn.Module,
    filepath: str | Path = "model.pth.tar",
    epoch_n: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
):
    """
    save_model(model)

    Save model.

    Args:
    - model (torch.nn.Module): Torch model
    - filepath (str, optional): Path at which to store model.
        Default is "model.pth.tar".
    - epoch_n (int, optional): Current epoch number. Default is 0.
    - optimizer (torch.optim.Optimizer, optional): Torch optimizer. Default is None.
    """

    state_dict = {
        "epoch_n": epoch_n,
        "net": "PredHPC",
        "state_dict": model.state_dict(),
    }

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()

    torch.save(state_dict, str(filepath))


def load_model(model, filepath: str | Path = "model.pth.tar") -> dict:
    """
    load_model(model)

    Load model from path.

    Args:
    - model (torch.nn.Module): Torch model
    - filepath (str or Path, optional): Path to the stored model. Default is
        "model.pth.tar".

    Raises:
    - OSError: If filepath doesn't exist.

    Returns:
    - checkpoint (dict): Checkpoint dictionary with model stored under "state_dict".
    """

    if not Path(filepath).is_file():
        raise OSError(f"'{filepath}' does not exist.")

    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    return checkpoint


def run_train(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    num_epochs: int = 100,
    device: str = "cpu",
    log_freq: int = 10,
    filepath: str | Path | None = None,
) -> dict[str, list]:
    """
    run_train(model, train_dl, val_dl)

    Runs training epochs.

    Args:
    - model (torch.nn.Module): Model.
    - train_dl (torch.util.data.DataLoader): Training dataloader.
    - val_dl (torch.util.data.DataLoader): Validation dataloader.
    - num_epochs (int, optional): Number of training epochs. Default is 100.
    - device (str, optional): Device to train on. Default is "cpu".
    - log_freq (int, optional): Logging frequency. Default is 10.
    - filepath (str or Path, optional): Path to load model and resume from,
        if applicable. Default is None.

    Returns:
    - history (dict): Dictionary with keys and values:
        - "epoch_n" (list): Epoch numbers.
        - "train_loss" (list): Train loss for each epoch.
        - "val_loss" (list): Validation loss for each epoch.
    """

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    start_epoch = 0
    if filepath is not None and Path(filepath).is_file():
        checkpoint = load_model(filepath)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch_n"]

    model.to(device)

    history = {
        key: [] for key in ["epoch_n", "train_loss", "val_loss"]
    }  # type: dict[str, list[int | np.float64]]
    epoch_n = -1
    for epoch_n in range(start_epoch, num_epochs + 1):
        model.train()
        history["epoch_n"].append(epoch_n)

        epoch_losses = list()
        for i, (X, y) in enumerate(train_dl):
            pred = model(X.to(device))

            # evaluate loss
            loss = criterion(pred, y)
            epoch_losses.append(loss.detach() / len(X))

            if epoch_n == 0:  # skip training for epoch 0
                break

            # backprop loss
            optimizer.zero_grad()
            loss.backward()

            # step optimizer
            optimizer.step()
            model.zero_grad()

        history["train_loss"].append(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            epoch_losses = list()
            for i, (X, y) in enumerate(val_dl):
                pred = model(X.to(device))

                # evaluate loss and MAE
                loss = criterion(pred, y).detach()
                epoch_losses.append(loss / len(X))

            history["val_loss"].append(np.mean(epoch_losses))

        # occasionally log to console
        if not epoch_n % log_freq:
            val_loss = history["val_loss"][-1]
            print(f"Epoch {epoch_n}/{num_epochs}: MSE={val_loss:.4f}")

    if filepath is not None:
        if Path(filepath).is_file():
            print(f"Overwriting {filepath}...")
            time.sleep(5)
        save_model(model, filepath, epoch_n, optimizer)

    return history


def predict(
    model: torch.nn.Module, X: torch.Tensor
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    predict(model, X)

    Predict from inputs, using provided model.

    Args:
    - model (torch.nn.Module): Model.
    - X (2D Tensor): Model input with shape (batch_size, input_size).

    Returns:
    - prediction (2D Tensor): Predicted outputs with shape (batch_size, pred_size).
    """

    model.eval()
    with torch.no_grad():
        prediction = model(torch.Tensor(X))

    return prediction.to("cpu").detach().numpy()


def get_prediction_step_idxs(
    n: int,
    num_prediction_steps: int = 1,
    prop_val: float = 0.2,
    ordered: bool = True,
    minimum_num_examples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    get_prediction_step_idxs(n)

    Obtain indices of training and validation sets for a specific number of prediction
    steps.

    Args:
    - n (int): Number of examples.
    - num_prediction_steps (int, optional): Number of prediction steps for which to
        get indices. Default is 1.
    - prop_val (float, optional): Proportion of examples to use for validation set.
        Default is 0.2.
    - ordered (bool, optional): Whether to use only the last examples for the
        validation set. Default is True.
    - minimum_num_examples (int, optional): Minimum number of examples to use for
        validation set.

    Raises:
    - ValueError: If validation set has fewer than minimum_num_examples examples.

    Returns:
    - training_idxs (torch.IntTensor): Indices for training set.
    - training_idxs_y (torch.IntTensor): Indices for training set, for y.
    - validation_idxs (torch.IntTensor): Indices for validation set
    - validation_idxs_y (torch.IntTensor): Indices for validation set, for y.
    """

    if num_prediction_steps == 0:
        training_idxs_np, validation_idxs_np = get_indices(n, prop_val, ordered=ordered)
        training_idxs = torch.from_numpy(training_idxs_np)
        validation_idxs = torch.from_numpy(validation_idxs_np)
        training_idxs_y, validation_idxs_y = training_idxs, validation_idxs
    else:
        gap = num_prediction_steps * 2
        base_idxs_x = np.arange(n // gap) * gap

        idxs_x = np.sort(
            np.concatenate(
                [np.int64(base_idxs_x + 1) for i in range(num_prediction_steps)]
            )
        )
        idxs_y = idxs_x + num_prediction_steps
        n = len(idxs_x)

        if ordered:
            all_idxs_x = idxs_x
            all_idxs_y = idxs_y
        else:
            shuffle_idx = np.arange(n)
            np.random.shuffle(shuffle_idx)
            all_idxs_x = idxs_x[shuffle_idx]
            all_idxs_y = idxs_y[shuffle_idx]

        num_val = int(n * prop_val)
        training_sub_idxs = np.arange(n - num_val)
        validation_sub_idxs = np.arange(n - num_val, n)  # final indices

        training_idxs = torch.from_numpy(all_idxs_x[training_sub_idxs])
        validation_idxs = torch.from_numpy(all_idxs_x[validation_sub_idxs])
        training_idxs_y = torch.from_numpy(all_idxs_y[training_sub_idxs])
        validation_idxs_y = torch.from_numpy(all_idxs_y[validation_sub_idxs])

    if len(validation_idxs) < minimum_num_examples:
        raise ValueError(
            f"Validation set has {len(validation_idxs)} examples, but requires  "
            f"at least {minimum_num_examples} examples."
        )

    return training_idxs, training_idxs_y, validation_idxs, validation_idxs_y


def get_indices(n: int, prop_val: float = 0.2, ordered: bool = True) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int], np.dtype[np.int64]],
]:
    """
    get_indices(n)

    Obtain indices for training and validation sets.

    Args:
    - n (int): Number of examples for which to obtain indices.
    - prop_val (float, optional): Proportion of examples to use for validation set.
        Default is 0.2.
    - ordered (bool, optional): Whether to use only the last examples for the
        validation set. Default is True.

    Returns:
    - training_idxs (1D np.ndarray): Indices for training set.
    - validation_idxs (1D np.ndarray): Indices for validation set.
    """

    num_val = int(n * prop_val)
    if ordered:
        val_idx = np.arange(n - num_val, n)  # final indices
    else:
        val_idx = np.sort(np.random.choice(n, num_val, replace=False))

    validation_mask = np.zeros(n, dtype=bool)
    validation_mask[val_idx] = True

    training_idxs = np.arange(n)[~validation_mask]
    validation_idxs = np.arange(n)[validation_mask]

    return training_idxs, validation_idxs


def get_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor | None = None,
    num_prediction_steps: int = 0,
    batch_size: int = 32,
    prop_val: float = 0.2,
    ordered: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    get_dataloaders(X)

    Obtain dataloaders from input data.

    Args:
    - X (2D Tensor): Model input with shape (num_examples, input_size).
    - y (2D Tensor, optional): Model target with shape (num_examples, output_size).
        If None, no y data is included in the dataloaders. Default is None.
    - num_prediction_steps (int, optional): Number of steps ahead to predict, if y is
        None. Default is 0.
    - batch_size (int, optional): Batch size. Default is 32.
    - prop_val (float, optional): Validation proportion. Default is 0.2.
    - ordered (bool, optional): If True, data is not shuffled.
        Default is True.

    Raises:
    - ValueError: If X and y do not have the same length.

    Returns:
    - train_dl (torch.util.data.DataLoader): Training dataloader.
    - val_dl (torch.util.data.DataLoader): Validation dataloader.
    """

    if y is not None:
        if num_prediction_steps != 0:
            raise ValueError("`num_prediction_steps` must be 0 if y is not None.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        training_idxs_np, validation_idxs_np = get_indices(len(X), prop_val, ordered)
        training_idxs = torch.from_numpy(training_idxs_np)
        validation_idxs = torch.from_numpy(validation_idxs_np)
        training_idxs_y = training_idxs
        validation_idxs_y = validation_idxs
    else:
        (
            training_idxs,
            validation_idxs,
            training_idxs_y,
            validation_idxs_y,
        ) = get_prediction_step_idxs(
            len(X), num_prediction_steps, prop_val, ordered=ordered
        )

    use_y = X if y is None else y

    training_dataloader = TensorDataset(
        torch.Tensor(X[training_idxs]), torch.Tensor(use_y[training_idxs_y])
    )
    validation_dataloader = TensorDataset(
        torch.Tensor(X[validation_idxs]), torch.Tensor(use_y[validation_idxs_y])
    )

    train_dl = DataLoader(
        training_dataloader,
        batch_size=batch_size,
        drop_last=False,
        shuffle=not (ordered),
    )

    val_dl = DataLoader(
        validation_dataloader,
        batch_size=batch_size,
        drop_last=False,
        shuffle=not (ordered),
    )

    return train_dl, val_dl
