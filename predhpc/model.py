from pathlib import Path
import time

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary as tsummary


class PredHPC(nn.Module):
    
    def __init__(self, input_size=300, n_DG_CA3=100, n_CA1=100, pred_size=2, 
                 summary=True):
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
        self.input_to_DG_CA3 = nn.Linear(self.input_size, self.n_DG_CA3)
        self.DG_CA3_to_CA1_soma = nn.Linear(self.n_DG_CA3, self.n_CA1)

        # convergence
        # self.CA1_convergence

        # direct path
        self.input_to_CA1_dend = nn.Linear(self.input_size, self.n_CA1)
        self.CA1_dend_decoder = nn.Linear(self.n_CA1, self.pred_size)

        if summary:
            tsummary(self, (input_size, ))


    def forward(self, x):
        """Forward definition

        Args:
            x (2D Torch tensor): input activations (batch_size x input_size).

        Returns:
            pred (2D Torch tensor): model predictions (batch_size x pred_size).
        """
        
        ## long path
        # self.DG_CA3 = F.relu(self.input_to_DG_CA3(x))
        # self.CA1_soma = F.relu(self.DG_CA3_to_CA1_soma(self.DG_CA3)) # dead-end for now

        # direct path
        self.CA1_dend = F.relu(self.input_to_CA1(x))

        # convergence
        self.CA1 = self.CA1_dend

        # prediction
        pred = F.relu(self.CA1_dend_decoder(self.CA1))

        return pred



def save_model(model, filepath="model.pth.tar", epoch_n=0, optimizer=None):
    """Saves model.

    Args:
        model (torch.nn.Module): torch model
        filepath (str, optional): Path at which to store model. 
            Defaults to "model.pth.tar".
        epoch_n (int, optional): Current epoch number. Defaults to 0.
        optimizer (torch.optim, optional): _description_. Defaults to None.
    """

    state_dict = {
        "epoch_n": epoch_n,
        "net": "PredHPC",
        "state_dict": model.state_dict(),
    }

    if optimizer is not None:
        state_dict["optimizer"]: optimizer.state_dict()

    torch.save(state_dict, str(filepath)) 


def load_model(model, filepath="model.pth.tar"):
    """Loads model from path.

    Args:
        model (torch.nn.Module): torch model
        filepath (str, optional): Path to the stored model. Defaults to 
            "model.pth.tar".

    Raises:
        OSError: Filepath doesn't exist

    Returns:
        checkpoint (): loaded model
    """

    
    if not Path(filepath).is_file():
        raise OSError(f"'{filepath}' does not exist.")

    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    return checkpoint
        

def run_train(model, train_dl, val_dl, num_epochs=100, device="cpu", 
              log_freq=10, filepath=None):
    """Runs training epochs.

    Args:
        model (torch.nn.Module): Model.
        train_dl (torch.utils.data.DataLoader): Training dataloader.
        val_dl (torch.utils.data.DataLoader): Validation dataloader.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        device (str, optional): Device to train on. Defaults to "cpu".
        log_freq (int, optional): Logging frequency. Defaults to 10.
        filepath (str or Path, optional): Path to load model and resume from, 
            if applicable. Defaults to None.

    Returns:
        history (dict): Dictionary storing 'epoch_n', 'train_loss' and 
            'val_loss' lists.
    """

    criterion = nn.MSELoss()

    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    start_epoch = 0
    if filepath is not None and Path(filepath).is_file():
        checkpoint = model.load(filepath)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch_n"]

    model.to(device)

    history = {key: [] for key in ["epoch_n", "train_loss", "val_loss"]}    
    for epoch_n in range(start_epoch, num_epochs + 1):
        model.train()
        history["epoch_n"].append(epoch_n)

        epoch_losses = []
        for i, (X, y) in enumerate(train_dl):
            pred = model(X.to(device))

            # evaluate loss
            loss = criterion(pred, y)
            epoch_losses.append(loss.detach() / len(X))
            
            if epoch_n == 0: # skip training for epoch 0
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
            epoch_losses = []
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
        model.save_model(filepath, epoch_n, optimizer)

    return history


def predict(model, X):
    """Predict from inputs, using the model.

    Args:
        model (torch.nn.Module): Model.
        X (2D Tensor): Model input (batch_size x input_size).

    Returns:
        pred (2D Tensor): Predicted outputs (batch_size x pred_size).
    """

    model.eval()
    with torch.no_grad():
        pred = model(torch.Tensor(X))
    
    return pred.to("cpu").detach().numpy()



def get_dls(X, y=None, batch_size=32, prop_val=0.2, ordered=True):
    """Returns dataloader built from the input data.

    Args:
        X (2D Tensor): Model input (num_examples x input_size).
        y (2D Tensor, optional): Model target (num_examples x output_size). 
            Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 32.
        prop_val (float, optional): Validation proportion. Defaults to 0.2.
        ordered (bool, optional): If True, data is not shuffled. 
            Defaults to True.

    Raises:
        ValueError: If X and y do not have the same length

    Returns:
        train_dl (torch.utils.data.DataLoader): Training dataloader.
        val_dl (torch.utils.data.DataLoader): Validation dataloader.
    """

    if y is not None and len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    num_val = int(len(X) * prop_val)
    if ordered:
        val_idx = np.arange(len(X) - num_val, len(X)) # final indices
    else:
        val_idx = np.sort(np.random.choice(len(X), num_val, replace=False))

    val_mask = np.zeros(len(X), dtype=bool)
    val_mask[val_idx] = True
    train_mask = ~val_mask

    X_train = TensorDataset(torch.Tensor(X[train_mask]))
    X_val = TensorDataset(torch.Tensor(X[val_mask])) 

    y_train, y_val = None, None
    if y is not None:
        y_train = TensorDataset(torch.Tensor(X[train_mask]))
        y_val = TensorDataset(torch.Tensor(X[val_mask]))

    train_dl = DataLoader(
        X_train, 
        y_train, 
        batch_size=batch_size, 
        drop_last=False,
        shuffle=not(ordered)
        )

    val_dl = DataLoader(
        X_val,
        y_val,
        batch_size=batch_size,
        drop_last=False,
        shuffle=not(ordered)
        )

    return train_dl, val_dl

