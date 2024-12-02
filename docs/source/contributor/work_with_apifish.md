# How work with apiFISH

## Recovery of Big-FISH updates
*Procedure to integrate Big-FISH updates into apiFISH*

### 1. Connect big-fish repository with apifish
- Find a remote name to call big-fish repository with an alias.
- Execute this command into your local apifish repository:
    `git remote add <remote-name> <big-fish git URL>`

### 2. Upload the required data from big-fish
- Find the branch name where you want to collect some commits
- Execute this command:
	`git fetch <remote-name> <branch-name>`

### 3. Merge selected commit into apifish
- Find the commit identifier on Big-FISH repository or with `git log`
- Execute this command from your apifish branch:
	`git cherry-pick <commit-ID>`

- Fix potential conflicts
- Run `git cherry-pick --continue`
- You can push and continue developments

### Remove the big-fish remote
- Remove the remote-name. All remote-tracking branches and configuration settings for the remote are removed.
- `git remote remove <remote-name>`
