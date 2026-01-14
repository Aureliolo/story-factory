# GitHub Copilot PR Auto-Run Setup

To enable GitHub Actions to automatically run on Copilot-generated pull requests without manual approval, you need to configure the repository settings.

## Steps to Enable Auto-Run

1. Go to your repository settings: `https://github.com/Aureliolo/story-factory/settings`

2. Navigate to **Actions** → **General** (in the left sidebar)

3. Scroll down to **Fork pull request workflows from outside collaborators**

4. Select one of these options:
   - **"Require approval for first-time contributors who are new to GitHub"** - Recommended for most repos
   - **"Require approval for first-time contributors"** - More permissive
   - **"Require approval for all outside collaborators"** - Most restrictive

5. For GitHub Copilot specifically, you may also need to:
   - Navigate to **Settings** → **Code security and analysis**
   - Enable **"Allow GitHub Actions to create and approve pull requests"**

## Why This Is Needed

GitHub Actions workflows triggered by `pull_request` events from bots (including GitHub Copilot) may require approval depending on your repository settings. The workflow file itself cannot override these settings - they must be configured at the repository level.

## Security Considerations

The current workflow configuration uses:
- Standard `pull_request` trigger (safe - runs in context of PR)
- Limited permissions (`contents: read`, `pull-requests: read`)
- Only read operations (checkout, install, test, lint)

This is secure for automated PRs from trusted sources like GitHub Copilot.

## Alternative: Manual Approval

If you prefer to keep manual approval for all bot PRs, you can:
1. Keep the current repository settings
2. Manually approve workflow runs from the Actions tab for each Copilot PR
