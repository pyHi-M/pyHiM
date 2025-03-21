# Developer installation

## Clone *pyHiM* repository

1. Create a folder where you want to install *pyHiM* and go inside to clone the repository. Standard location to do it is: ```$HOME/Repositories/pyHiM```

```bash
mkdir $HOME/Repositories
cd $HOME/Repositories
```

2. Choose your clone method between HTTPS or SSH key:
    - HTTPS
      ```bash
      git clone https://github.com/marcnol/pyHiM.git
      ```
    - SSH
      ```bash
      git clone git@github.com:marcnol/pyHiM.git
      ```

3. Open your $HOME/.bashrc using nano

```bash
nano $HOME/.bashrc
```

4. Add the following line to the end

```sh
export PATH="$PATH:$HOME/Repositories/repo-marcnol/pyHiM/src"
export PATH="$PATH:$HOME/Repositories/repo-marcnol/pyHiM/src/toolbox/file_handling"
export PATH="$PATH:$HOME/Repositories/repo-marcnol/pyHiM/src/postProcessing"

export PYTHONPATH="$PYTHONPATH:$HOME/Repositories/repo-marcnol/pyHiM/src"
export MPLBACKEND=agg
```

```{note}
Make sure you change ```.../Repositories/...``` with your directory name (step 1.) if this is not where you put *pyHiM* !
```

## Install conda

Follow the Miniconda instructions:
https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install

## Automatically configure pyHiM

Run this command in your terminal:

```sh
conda env create -f environment.yml
```

```{note}
If you get this error:
`ImportError: Dask\'s distributed scheduler is not installed.`

You solve by running `pip install dask[complete] distributed --upgrade`.
```

## Install apifish module

1. Navigate where you want install apifish
```bash
cd $HOME/Repositories
```

2. Choose your clone method between HTTPS or SSH key:
    - HTTPS
      ```bash
      git clone https://github.com/apiFISH/apiFISH.git
      ```
    - SSH
      ```bash
      git clone git@github.com:apiFISH/apiFISH.git
      ```
3. Switch on `development` branch
```bash
cd apiFISH && git checkout development
```

4. Update `PYTHONPATH` env variable by adding the following line to your local ~/.bashrc

```sh
export PYTHONPATH="$PYTHONPATH:$HOME/Repositories/apiFISH"
```

## Test pyHiM

- The tests use the `pytest` module.
- The test resources are inside `pyhim-small-dataset`. It's a sub-module of pyHiM, so to get the dataset you need to run:
  * `git submodule update --init --recursive`
  OR
  * `git clone --recurse-submodules <HTTPS/SSH>`

- To run the tests:

```bash
cd ~Repositories/pyHiM/
conda activate pyhiM39
pytest tests/ -vv
```

## Additional installation to generate documentation

```sh
conda install sphinx
conda install -c conda-forge myst-parser
conda install -c conda-forge sphinxcontrib-mermaid
conda install -c conda-forge sphinx-panels
conda install -c conda-forge sphinx_rtd_theme
```
Update `PYTHONPATH` env variable, for fileProcessing scripts documentation, by adding the following line to your local ~/.bashrc

```sh
export PYTHONPATH="$PYTHONPATH:$HOME/Repositories/pyHiM/src/fileProcessing"
```

## Build documentation locally
Install in your conda env:
```bash
pip install nbsphinx ipython sphinx-book-theme
conda install pandoc
```
Generate documentation:
```bash
cd docs/
make html
```
A `build/html/` folder has been created with a `index.html` file inside, open it with your favorite browser.

## Script installation for super-computer centers (e.g. Meso-LR)

To access the private repository of *pyHiM*, please first create an SSH key and put it in your keyring. Follow the steps described [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

Then run the following automatic script:

```sh
#!/bin/bash

# load conda
module load  python/Anaconda/3-5.1.0

# create environment and install packages
conda create --name pyHiM python=3.7.2 dask numpy matplotlib astropy scikit-learn pandas
conda activate pyHiM
conda install photutils -c astropy
pip install mrc roipoly tqdm stardist csbdeep pympler
pip install --upgrade tensorflow

# api-fish
cd $HOME/Repositories
git clone git@github.com:apiFISH/apiFISH.git
cd apifish && git checkout development
echo 'export PYTHONPATH="$PYTHONPATH:$HOME/Repositories/apiFISH"'  >> ~/.bashrc

# clone pyHiM
cd $HOME/Repositories
git clone git@github.com:marcnol/pyHiM.git
git checkout development

# settings
ln -s $HOME/Repositories/pyHiM/src/toolbox/file_handling/cleanHiM_run.py $HOME/bin/cleanHiM

```

## Step to setup pre-commit in local
- environment installation
  `pip install pre-commit`

- check if it's well installed
  `pre-commit --version`

- install command of the file ".pre-commit-config.yaml" inside ".git/hooks/pre-commit"
  `pre-commit install`

- fix strange issue or warning
  `pre-commit autoupdate --repo https://github.com/pre-commit/pre-commit-hooks`

- test pre-commit without any commit
  `pre-commit run --all-files`

- Update pre-commit file
  - `pre-commit clean`
  - `pre-commit autoupdate`
  - `pre-commit install`
