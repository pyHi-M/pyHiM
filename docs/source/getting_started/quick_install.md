# Installation

|OS|Linux|Windows|Mac|
|:-:|:-:|:-:|:-:|
|**compatibility**|Yes|Yes|Yes|

## Install conda

*We use conda environment to avoid version problem between pyHiM dependencies and other applications.*

We recommend to download the lighter version `miniconda` if you only intend to use pyHiM without developing new applications.

- [Installing conda on Linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [Installing conda on Windows](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)
- [Installing conda on macOS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

## Create conda environment

Open a **terminal** (for Windows user: from the Start menu, open the **Anaconda Prompt**). Create a conda environment and activate it:
```
conda create -n pyHiM python=3.9
conda activate pyHiM
```

## Install pyHiM

```bash
pip install pyhim
```

```{note}
To check if pyHiM is well installed, run:
`pyhim --help`
```
