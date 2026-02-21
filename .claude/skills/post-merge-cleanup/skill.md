---
description: "Post-merge cleanup: switch to default branch, pull, delete merged branches, prune remotes"
argument-hint: "[no arguments]"
allowed-tools:
  - Bash
---

# Post-Merge Cleanup

Switch to the default branch, pull latest, delete all merged local branches, and prune stale remote tracking refs.

## Steps

1. **Detect default branch, switch to it, and pull**

```bash
default_branch=$(git symbolic-ref --short refs/remotes/origin/HEAD | sed 's@^origin/@@') && git switch "$default_branch" && git pull --ff-only
```

2. **Delete merged local branches**

Delete every local branch that has been merged into the default branch (excluding it):

```bash
git branch --merged "$default_branch" | while read -r branch; do
  branch=$(echo "$branch" | sed 's/^[ *]*//')
  if [ -n "$branch" ] && [ "$branch" != "$default_branch" ]; then
    git branch -d "$branch"
  fi
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
