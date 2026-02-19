---
description: "Full PR review pipeline: local agents + external feedback + triage + implement fixes"
argument-hint: "[PR number, or blank for current branch]"
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Task
  - AskUserQuestion
---

# Aurelio PR Review

Full PR review pipeline that runs local review agents, fetches external reviewer feedback, triages everything, and implements approved fixes.

**Arguments:** "$ARGUMENTS"

---

## Phase 1: Find the PR

If an argument was provided, use it as the PR number. Otherwise, detect the current branch's PR:

```bash
gh pr list --head $(git branch --show-current) --json number,title --jq '.[0]'
```

Get the OWNER/REPO from:

```bash
gh repo view --json nameWithOwner -q .nameWithOwner
```

If no PR is found, ask the user for a PR number using AskUserQuestion.

## Phase 2: Run local review agents

Identify changed files and their types:

```bash
# If PR exists, diff against base branch
gh pr diff NUMBER --name-only

# Otherwise, diff against main
git diff main --name-only
```

Based on changed files, launch applicable review agents **in parallel** using the Task tool. **Do NOT use `run_in_background`** — launch them as regular parallel Task calls so results arrive together and the user sees all agents complete before triage begins. Background agents cause confusing late-arriving `task-notification` messages that make it look like you presented triage before agents finished.

| Agent | When to launch | subagent_type |
|---|---|---|
| **code-reviewer** | Always | `pr-review-toolkit:code-reviewer` |
| **pr-test-analyzer** | Test files changed | `pr-review-toolkit:pr-test-analyzer` |
| **silent-failure-hunter** | Error handling or try/except changed | `pr-review-toolkit:silent-failure-hunter` |
| **comment-analyzer** | Comments or docstrings changed | `pr-review-toolkit:comment-analyzer` |
| **type-design-analyzer** | Type annotations or classes added/modified | `pr-review-toolkit:type-design-analyzer` |

Each agent should receive the list of changed files and focus on reviewing them.

Collect all findings with their severity/confidence scores.

## Phase 3: Fetch external reviewer feedback

Fetch from three GitHub API sources **in parallel** using `gh api`:

1. **Review submissions** (top-level review bodies):

   ```bash
   gh api repos/OWNER/REPO/pulls/NUMBER/reviews --paginate
   ```

   Extract: author, state, body.

   **CRITICAL: Parse review bodies for outside-diff-range comments.** Some reviewers (e.g. CodeRabbit) embed actionable comments inside `<details>` blocks in the review body when the affected lines are outside the PR's diff range. Look for patterns like "Outside diff range comments (N)" and extract each embedded comment's file path, line range, severity, and description. These are just as important as inline comments — do NOT skip them.

2. **Inline review comments** (comments on specific lines):

   ```bash
   gh api repos/OWNER/REPO/pulls/NUMBER/comments --paginate
   ```

   Extract: author, file path, line number, body, subject_type.

3. **Issue-level comments** (general PR comments, e.g. CodeRabbit walkthrough):

   ```bash
   gh api repos/OWNER/REPO/issues/NUMBER/comments --paginate
   ```

   Extract: author, body (look for actionable items, not just summaries).

**Important:** Use `gh api` with `--jq` for filtering. Keep it simple and robust — no complex Python scripts to parse JSON.

**Note:** When review bodies are large (e.g., CodeRabbit's review with embedded outside-diff comments), fetch the **full body** without truncation. Use a projection that preserves all fields while limiting body length, e.g., `--jq '[.[] | {author: (.user.login // .author.login), state: .state, body: (.body | .[:15000])}]'` rather than slicing only the body (`.[].body | .[:15000]`), which discards author and state entirely. Note that jq's string slicing operates on Unicode code points (not bytes), so `.[:15000]` will never split a multi-byte character. Outside-diff comments are typically at the top of the review body.

## Phase 4: Consolidate and triage

**CRITICAL: Wait for ALL feedback sources before proceeding.** Do NOT present the triage table until every local review agent AND every external feedback fetch has completed. Since agents are launched as regular (non-background) parallel Task calls, their results arrive together in the same response — no need for `TaskOutput`. If any agent or fetch fails, retry it before proceeding. All agents must be confirmed complete before moving to triage.

Build a single consolidated table of ALL actionable feedback from both local agents and external reviewers.

For each item, determine:

- **Source**: Which agent or external reviewer found it
- **Severity**: Critical / Major / Medium / Minor
  - Local agent findings: map confidence 91-100 to Critical, 80-90 to Major, 60-79 to Medium, below 60 to Minor
  - External feedback: infer from reviewer labels if present, otherwise from context
- **File:Line**: Where the issue is
- **Issue**: One-line summary of the problem
- **Valid?**: Your assessment — is this correct advice for this codebase? Check against CLAUDE.md rules and actual code.

**Deduplication:** If multiple sources flag the same issue on the same line, merge into one item and note all sources.

**Conflict detection:** If two sources contradict each other, flag it and include both positions.

## Phase 5: Present for approval

Show the user the complete table, organized by severity (Critical first, Minor last). Include:

- Total count of items
- Count by source (each agent + each external reviewer)
- Any items you recommend skipping (with reasoning)

Then ask the user using AskUserQuestion with options like:

- "Implement all" (Recommended)
- "Let me review the list first"
- "Skip some items"

If the user wants to skip items, ask which ones by number.

## Phase 6: Implement fixes

For each approved item, grouped by file (to minimize context switches):

1. Read the file
2. Make the fix
3. Move to the next fix in the same file before switching files

After all fixes:
1. Run `ruff format .` and `ruff check --fix .`
2. If any fix changes test expectations (e.g. behavior change), update the affected tests
3. Only run targeted tests (1-2 test runs max) — focus on files that were directly modified, not the entire suite; rely on pre-push hooks and CI for full coverage

## Phase 7: Commit and push

After all fixes pass linting and tests:

1. Stage all modified files (specific files, not `git add .`)
2. Commit with a descriptive message summarizing what was fixed (e.g. "fix: address 28 PR review items from local agents, CodeRabbit, and Copilot")
3. Push to the current branch
4. If commit or push fails due to hooks, fix the actual issue and create a NEW commit — NEVER use `--no-verify` or `--amend`

## Phase 8: Verify external reviewer status

After pushing, check if external reviewers (especially CodeRabbit) have posted updated feedback on the new commits:

```bash
# Check for new reviews/comments since the push
gh api repos/OWNER/REPO/pulls/NUMBER/reviews --paginate
gh api repos/OWNER/REPO/pulls/NUMBER/comments --paginate
gh api repos/OWNER/REPO/issues/NUMBER/comments --paginate
```

**CodeRabbit pre-merge checks:** CodeRabbit's main issue-level comment (the walkthrough) often contains a status summary or "Actionable comments posted: N" count in its review bodies. After each review round, check:
1. Look at each CodeRabbit review body for "Actionable comments posted: N" — if N > 0, those comments need to be addressed.
2. Check for any new inline comments from CodeRabbit (or other reviewers) on the latest commit range.
3. If there are new actionable items that weren't in the original triage, address them or flag them to the user.

The goal is to ensure all external reviewer feedback is resolved before considering the PR review complete — not just the feedback from the first round.

## Phase 9: Summary

Report what was done:

- Number of items fixed (broken down by source)
- Files modified
- Tests passed/failed
- Any items that couldn't be fixed (with explanation)

---

## Rules

- Never skip a fix without telling the user why.
- If a fix requires changing tests, change the tests too.
- If a fix introduces new code paths, add test coverage.
- Group file edits to minimize re-reading files.
- Respect all rules in CLAUDE.md (formatting, logging, no placeholders, etc.).
- If two sources contradict each other, flag it and ask the user.
- Do NOT use `--no-verify` or `--amend` for commits.
- External feedback fetch failures are non-fatal — if the PR has no external reviews yet, proceed with local agent findings only.
- **Fix everything in the current PR — never defer.** Every valid recommendation must be implemented in this PR regardless of size. No creating GitHub issues for "too large" items, no deferring to future PRs, no marking things as out of scope. If a reviewer flags it and it's valid, fix it now — docstrings, type hints, refactors, all of it.
