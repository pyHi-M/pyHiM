# Reporting bugs and feature requests

The issue tracking system is available on GitHub in the [marcnol/pyHiM repository](https://github.com/marcnol/pyHiM/issues).

## For documentation reviewer

### 1. Clone *pyHiM* repository

Create a folder where you want to install *pyHiM* and go inside to clone the repository. Standard location to do it is: ```$HOME/Repositories/pyHiM```

```bash
mkdir $HOME/Repositories
cd $HOME/Repositories
```

Choose your clone method between HTTPS or SSH key:
- HTTPS
    ```bash
    git clone https://github.com/marcnol/pyHiM.git
    ```
- SSH
    ```bash
    git clone git@github.com:marcnol/pyHiM.git
    ```

### 2. Switch on the documentation branch

For pyHiM version 0.6.0, the online documentation is based on `development` branch:

```shell
git checkout development
```

Don't forget to update your local `development` branch with remote `development`:

```shell
git pull
```

### 3. Create a new branch for your modification

Create and switch on this new branch:
```shell
git checkout -b doc/name_of_my_branch
```

### 4. Fix what you want

You're reading the [online documentation](https://pyhim.readthedocs.io/en/latest/index.html) and you find something to fix:

- Check your web link like https://pyhim.readthedocs.io/en/latest/**user_guide/pyhim_presentation**.html
- With your file editor, go to `pyHiM` > `docs` > `source` > **user_guide > pyhim_presentation**.md
- Fix what you want

### 5. Share your modifications

- `git add -A`
- `git commit -m "write your message here"`
- `git push`
- Create a pull request [on github](https://github.com/marcnol/pyHiM/pulls)
- wait for a developer to validate the PR

### THANKS!
