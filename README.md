# Few-shot learning of predictive features with distal apical dendrites in the hippocampus

## 1. Description

This repository contains code to simulate few-shot learning of predictive features with dendrites and behavioural timescale synaptic plasticity (BTSP) in a hippocampal circuit. See our preprint on _bioRxiv_. 

Neural activity is simulated in a circuit composed of place cells, object cells, two-compartment pyramidal neurons and inhibitory interneurons while an agent visits target landmark objects in a continuous linear track or open field environment.  

![Schematic of model](https://drive.google.com/uc?export=view&id=1NS5e5PGoWfhhvniYncwuWGv7gvfxmnST)

The simulations were developed using the [`RatInABox`](https://github.com/RatInABox-Lab/RatInABox) package.

## 2. Installation

This package can be installed, optionally in a virtual environment, using `pip install git+https://github.com/colleenjg/predhpc` and imported with `import predhpc`.

This code has been tested with `Python 3.11`. For package dependencies, see `requirements.txt`.

## 3. Scripts and modules

- **`predhpc/`**:
    - `env`: Custom `RatInABox` environments (e.g., `LinearResetEnv`, `TEnv`, `OpenField`).
    - `agent`: Custom `RatInABox` agents (e.g., `ResetableAgent`, `LinearResetAgent`, `TAgent`, `OpenFieldAgent`).
    - **`neurons/`**: Custom `RatInABox` neuron layers (e.g., `ObjectCells`, `BTSPLayer`, `NMDALayer`, `TwoCompLayer`).
    - **`experiments/`**: Linear track analyses and metrics.
    - **`util/`**: Custom utilities.
    - `run_manager`: Tools for running simulations in various environments.
- **`scripts/`**: Examples of simulations and analyses.
- **`results/`**: Results from script, paper and experiment analyses.

## 4. Paper

The code for running the analyses reported in the paper, and for reproducing the figures can be found under `predhpc/paper` and `predhpc/paper_plot_fcts`.   
Paper figures are also reproduced in `predhpc/scripts/paper.ipynb`.

## 5. Author

Code written by Colleen Gillon (c _dot_ gillon _at_ imperial _dot_ ac _dot_ uk).

Please do not hesitate to contact me or open an issue/pull request, if you have trouble using the codebase or have improvements to propose.  