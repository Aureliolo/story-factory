# GitHub Copilot Instructions

This repository is configured with custom instructions for GitHub Copilot coding agent to provide better guidance when making code changes.

## Overview

GitHub Copilot can be assigned issues and will create pull requests to address them. To ensure Copilot produces high-quality, consistent code, we've configured custom instructions that guide its behavior.

## Instruction Files

### Repository-Wide Instructions

**`.github/copilot-instructions.md`**
- Applies to all tasks assigned to Copilot in this repository
- Contains project overview, architecture, and general coding standards
- Key sections:
  - Critical Rules (no placeholders, complete implementations, logging, testing)
  - Code Standards (Python best practices, testing requirements, development flow)
  - Repository Structure (detailed file/directory layout)
  - Architecture & Key Patterns (Service Container, dependency injection, error handling)
  - Agent Workflow (how the multi-agent system works)
  - Key Guidelines (Ollama integration, state management, configuration)

### Path-Specific Instructions

Located in `.github/instructions/`, these provide specialized guidance for specific file types:

**`test-files.instructions.md`** (`applyTo: "**/test_*.py"`)
- Guidelines for writing and modifying test files
- Test structure and organization patterns
- Coverage requirements (100% for core modules)
- Mocking guidelines (always mock Ollama API)
- Example test patterns and fixtures
- Applies to all `test_*.py` files in any directory

**`ui-pages.instructions.md`** (`applyTo: "ui/pages/*.py"`)
- NiceGUI page component requirements
- Page structure with dependency injection
- UI element usage and best practices
- Async operation handling
- State management patterns
- Example page implementation
- Applies to all Python files in `ui/pages/` directory

**`agents.instructions.md`** (`applyTo: "agents/*.py"`)
- AI agent implementation guidelines
- Agent architecture (extending BaseAgent)
- Temperature settings per agent role
- Ollama integration patterns
- JSON parsing and validation
- Example agent implementation
- Applies to all Python files in `agents/` directory

## Benefits

These custom instructions help Copilot:

1. **Maintain Code Quality**: Enforces standards for formatting, linting, and testing
2. **Follow Architecture**: Respects the Service Container pattern and clean architecture
3. **Write Better Tests**: Ensures 100% coverage on core modules with proper mocking
4. **Integrate Properly**: Understands the multi-agent system and Ollama integration
5. **Use Correct Patterns**: Follows established patterns for error handling, logging, and state management

## How It Works

When Copilot is assigned an issue:

1. It reads `.github/copilot-instructions.md` for general guidance
2. When working on specific files, it also reads matching path-specific instructions
3. It uses these instructions to guide its code changes
4. It follows the development workflow (format, lint, test) before committing

## Best Practices

According to [GitHub's documentation](https://gh.io/copilot-coding-agent-tips), effective custom instructions should:

- ✅ **Be specific**: Our instructions include exact commands and patterns
- ✅ **Include examples**: We provide code examples for key patterns
- ✅ **Define workflows**: Clear development flow with build, test, and lint commands
- ✅ **Set standards**: Explicit code standards and quality requirements
- ✅ **Use path-specific instructions**: Specialized guidance for different file types

## Updating Instructions

When updating the custom instructions:

1. Edit the appropriate `.md` file in `.github/` or `.github/instructions/`
2. Keep instructions clear, specific, and actionable
3. Include examples where helpful
4. Test changes by assigning a simple issue to Copilot
5. Commit changes to update the instructions for future tasks

## References

- [GitHub Copilot Coding Agent Best Practices](https://gh.io/copilot-coding-agent-tips)
- [Adding Custom Instructions for GitHub Copilot](https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot)
