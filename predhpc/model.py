from typing import TYPE_CHECKING

import numpy as np
import torch


if TYPE_CHECKING:
    from predhpc.neurons import trainable_neurons


class TorchNeuronModel(torch.nn.Module):
    """
    TorchNeuronModel()

    Torch neuron model that updates weights of a ratinabox neuron layer.

    List of methods (in addition to torch.nn.Module methods):
        • self.set_device()
        • self.forward()
        • self.get_X()
        • self.get_y()
        • self.run_train()
        • self.run_eval()
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
