---
description: "Post-merge cleanup: switch to main, pull, delete merged branches, prune remotes"
allowed-tools:
  - Bash
---

# Post-Merge Cleanup

Switch to main, pull latest, delete all merged local branches, and prune stale remote tracking refs.

## Steps

1. **Switch to main and pull**

```bash
git checkout main && git pull
```

2. **Delete merged local branches**

Delete every local branch that has been merged into main (excluding main itself):

```bash
git branch --merged main | grep -v '^\*\|main$' | xargs -r git branch -d
```

If `xargs -r` isn't available (e.g. macOS), loop manually:

```bash
for branch in $(git branch --merged main | grep -v '^\*\|main$'); do
  git branch -d "$branch"
done
```

3. **Prune stale remote tracking branches**

```bash
git remote prune origin
```

4. **Report results**

Show the final state:

```bash
git branch -v
git log --oneline -3
```

Report what was deleted and the current branch status.

## Improvements for the future

- **Auto-detect unmerged stale branches**: flag local branches older than N days that haven't been merged — prompt the user before deleting.
- **Multi-remote support**: iterate over all remotes, not just `origin`.
- **Worktree cleanup**: detect and remove `.claude/worktrees/` leftover directories from abandoned worktree sessions.
- **Tag cleanup**: optionally prune local tags that no longer exist on the remote.
- **CI status check**: before deleting an unmerged branch, check if its last PR was closed without merge (using `gh pr list --state closed`) to distinguish abandoned branches from in-progress work.
