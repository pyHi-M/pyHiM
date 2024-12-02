# Development process

## Branching model
The following branches will always exist in *pyHiM*:

* The `master` branch: reference branch of stable versions. We use the tag [versioning](https://semver.org/) systems.
* The `development` branch: features stable but pre-release features. Use at your own risk.
* Features branches are used for development of single features
* Bug branches are use to fix specific bugs repported as `issues`. Once the bug is fixed, the branch is merged into the `development` branch using a pull request and subsequentially deleted.

Template for branch naming scheme:

```<context>/(<scope>/)short_definition_of_work```

- `<context>`:
    - `fix` for bug fix
    - `feature` for development about a new feature
    - `doc` for adding documentation
    - `refactor` for improvement of readability, reusability or structure
- `<scope>` is the feature ou field involved by this commit (optional)

Examples:
- ```doc/contributor/write_dev_process```
- ```feature/matching/spot_into_mask```

## Typical workflow using `git`

### Create a git branch

1. Create your new branch:
    ```shell
    git branch <new_branch>
    ```

2. Switch on the new branch:
    ```shell
    git checkout <new_branch>
    ```

3. Create it in origin to notify other users that you have created this branch:
    ```shell
    git push --set-upstream origin <new_branch>
    ```

4. You are ready to develop your new feature.

### Share new feature

1. Ensure that there isn't conflit on your branch
    ```shell
    git pull origin <your_branch>
    ```

2. Choose files you want to commit, to add all:

    ```shell
    git add -A
    ```

3. Commit your changes with a title with short description and a section with details:
    ```shell
    git commit -m "<type>(<scope>) - #<issue_number> Summary title" -m "- List of task or bugfix done in this commit"
    ```
    *See also [Good practices for commit](good_commit.md).*

4. Push your update
    ```shell
    git push origin <your_branch>
    ```

### Merge with development branch

The best way to validate your new features is to make a **pull request**. The code in the pull request will be reviewed by another member of the team before it can be merged into the Development branch.

1. On your GitHub repository, navigate to `pull requests` section
2. Button `New pull request`
3. Choose the both branch to compare
4. Before create this pull request, check your changes and check if your code is clean (without sauvage `print()` or commented code line)
5. Create pull request
6. Validate this pull request with another member of the team
7. Squash your commits with pull request option
8. Delete your feature branch locally and remotely (see below)
9. Start a new task !

### pyHiM releases

New release versions are made using [Semantic Versioning](https://semver.org/).

Given a version number MAJOR.MINOR.PATCH, increment the:

1. MAJOR version when you make incompatible API changes,
2. MINOR version when you add functionality in a backwards compatible manner, and
3. PATCH version when you make backwards compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

### show git activity history

```shell
git log --graph --oneline
```

### Delete your branch

After you merge your pull request, please delete your branch. This will signal to other developers that the work on the branch is complete and prevents you or others from accidentally using old branches. For more information, see "[Deleting and restoring branches in a pull request](https://docs.github.com/en/github/administering-a-repository/deleting-and-restoring-branches-in-a-pull-request)."

When you delete a branch, the commit history will be transfered to the development branch. You can always restore your deleted branch or revert your pull request if needed.

```shell
// delete branch locally
git branch -d local_branch_name

// delete branch remotely
git push origin --delete remote_branch_name
```

```{note}
If you get the error below, it may mean that someone else has already deleted the branch.

    error: unable to push to unqualified destination: remoteBranchName The destination refspec neither matches an existing ref on the remote nor begins with refs/, and we are unable to guess a prefix based on the source ref. error: failed to push some refs to 'git@repository_name'


Try to synchronize your branch list using:

    git fetch -p

The `-p` flag means `prune`. After fetching, branches which no longer exist on the remote will be deleted.
```

### Recover deleted branch

- [Deleting and restoring branches in a pull request](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/deleting-and-restoring-branches-in-a-pull-request)

-  [Recover deleted git branch from local](https://imran-ahmad.medium.com/how-to-recover-restore-deleted-git-branch-5a068c07bed2)

Find the `SHA` for the commit at the tip of your deleted branch using: Use `git reflog` to do so:

```shell
git reflog
```

To restore the branch, use:

```shell
git checkout -b <branch> <sha>
```

## Git squash
*If you want to squash working/insignificant commits from your <dev_branch>.*

0. You can store your branch history in an over-branch for your first time:
```bash
git checkout dev_branch
git pull dev_branch
git branch save/dev_branch
```

1. Get an overview of your latest commits using `git log` or the GitHub interface.

2. Identify the commit that you want to keep, and copy the hash of the previous (older) commit.
![hash selected](../_static/contributor/git-log-hash-selected-simplify.png)

3. Make an interactive rebase:
```bash
git rebase -i <older-commit-hash>
```

4. It's open a text editor (as image below), edit `pick` to `squash` for each commit that you want to squash within the top commit. Save, exit.
![Rebase example](../_static/contributor/rebase-example.png)

5. An other text editor are open with the list of commit messages, edit them. Save, exit.

6. Check if it's good with a new `git log` or with a `git diff <commit1> <commit2>`.

7. Apply changes on the remote repository:
```bash
git push -f
```

![Git squash end](../_static/contributor/git-squash-end.png)

8. Delete your save branch:
```bash
git branch -D save/dev_branch
```
