from pathlib import Path
from typing import TYPE_CHECKING
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary as tsummary

if TYPE_CHECKING:
    from predhpc import neurons


class TorchNeuronModel(torch.nn.Module):
    def __init__(
        self,
        seq_neuron_layers: list["neurons.TorchLayer"],
        device: str = "cpu",
        lr: float = 1e-4,
        RMSprop: bool = False,
    ):
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

    def set_device(self, device: str = "cpu"):
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        for layer in self.seq_layers:
            x = layer(x)
        return x

    def get_X(self, n):
        if len(self.seq_neuron_layers[0].history["firingrate"]) < n:
            raise ValueError(f"Fewer than {n} steps recorded.")

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
        if len(self.seq_neuron_layers[0].history["firingrate"]) < n:
            raise ValueError(f"Fewer than {n} steps recorded.")

        y = torch.Tensor(self.seq_neuron_layers[0].Agent.history["pos"][-n:])

        return y

    def run_train(self, n: int):
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


class PredHPC(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 300,
        n_DG_CA3: int = 100,
        n_CA1: int = 100,
        pred_size: int = 2,
        summary: bool = True,
    ):
        """_summary_

        Args:
            input_size (int, optional): Number of inputs to the network.
                Defaults to 300.
            n_DG_CA3 (int, optional): Size of the DG/CA3 layer.
                Defaults to 100.
            n_CA1 (int, optional): Size of the CA1 layer.
                Defaults to 100.
            pred_size (int, optional): Size of the predictive output layer.
                Defaults to 2.
            summary (bool, optional): If True, a summary of the model is
                printed to the console. Defaults to True.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward definition

        Args:
            x (2D Torch tensor): input activations (batch_size x input_size).

        Returns:
            prediction (2D Torch tensor): model predictions (batch_size x pred_size).
        """

        ## long path
        # self.DG_CA3 = F.relu(self.input_to_DG_CA3(x))
        # self.CA1_soma = F.relu(self.DG_CA3_to_CA1_soma(self.DG_CA3)) # dead-end for now

        # direct path
        self.CA1_dend = F.relu(self.input_to_CA1_dend(x))

        # convergence
        self.CA1 = self.CA1_dend

        # prediction
        prediction = F.relu(self.CA1_dend_decoder(self.CA1))

        return prediction


def save_model(
    model: torch.nn.Module,
    filepath: str | Path = "model.pth.tar",
    epoch_n: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
):
    """Saves model.

    Args:
        model (torch.nn.Module): torch model
        filepath (str, optional): Path at which to store model.
            Defaults to "model.pth.tar".
        epoch_n (int, optional): Current epoch number. Defaults to 0.
        optimizer (torch.optim.Optimizer, optional): Torch optimizer. Defaults to None.
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
    """Loads model from path.

    Args:
        model (torch.nn.Module): torch model
        filepath (str, optional): Path to the stored model. Defaults to
            "model.pth.tar".

    Raises:
        OSError: Filepath doesn't exist

    Returns:
        checkpoint (dict): checkpoint used to load model
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
    """Runs training epochs.

    Args:
        model (torch.nn.Module): Model.
        train_dl (torch.util.data.DataLoader): Training dataloader.
        val_dl (torch.util.data.DataLoader): Validation dataloader.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        device (str, optional): Device to train on. Defaults to "cpu".
        log_freq (int, optional): Logging frequency. Defaults to 10.
        filepath (str or Path, optional): Path to load model and resume from,
            if applicable. Defaults to None.

    Returns:
        history (dict): Dictionary storing 'epoch_n', 'train_loss' and
            'val_loss' lists.
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
    """Predict from inputs, using the model.

    Args:
        model (torch.nn.Module): Model.
        X (2D Tensor): Model input (batch_size x input_size).

    Returns:
        prediction (2D Tensor): Predicted outputs (batch_size x pred_size).
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
    """Returns indices for training and validation sets.

    Args:
        n (int): Number of examples.
        num_prediction_steps (int, optional): Number of prediction steps. Defaults to 1.
        prop_val (float, optional): Proportion of examples to use for validation set.
            Defaults to 0.2.
        ordered (bool, optional): Whether to use only the last examples for the
            validation set. Defaults to True.
        minimum_num_examples (int, optional): Minimum number of examples to use for
            validation set.

    Returns:
        training_idxs (torch.IntTensor): Indices for training set.
        training_idxs_y (torch.IntTensor): Indices for training set, for y.
        validation_idxs (torch.IntTensor): Indices for validation set
        validation_idxs_y (torch.IntTensor): Indices for validation set, for y.
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


def get_indices(
    n: int, prop_val: float = 0.2, ordered: bool = True
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int], np.dtype[np.int64]],
]:
    """Returns indices for training and validation sets.

    Args:
        n (int): Number of examples.
        prop_val (float, optional): Proportion of examples to use for validation set.
            Defaults to 0.2.
        ordered (bool, optional): Whether to use only the last examples for the
            validation set. Defaults to True.

    Returns:
        training_idxs (1D array): Indices for training set
        validation_idxs (1D array): Indices for validation set
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
    """Returns dataloaders built from the input data.

    Args:
        X (2D Tensor): Model input (num_examples x input_size).
        y (2D Tensor, optional): Model target (num_examples x output_size).
            Defaults to None.
        num_prediction_steps (int, optional): Number of steps ahead to predict, if y is
            None. Defaults to 0.
        batch_size (int, optional): Batch size. Defaults to 32.
        prop_val (float, optional): Validation proportion. Defaults to 0.2.
        ordered (bool, optional): If True, data is not shuffled.
            Defaults to True.

    Raises:
        ValueError: If X and y do not have the same length

    Returns:
        train_dl (torch.util.data.DataLoader): Training dataloader.
        val_dl (torch.util.data.DataLoader): Validation dataloader.
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
