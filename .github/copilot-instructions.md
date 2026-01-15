# Story Factory - Copilot Instructions

This is a Python-based multi-agent system for generating stories using local AI models via Ollama. The system uses specialized AI agents (Interviewer, Architect, Writer, Editor, Continuity Checker) working together to create short stories, novellas, and novels with iterative refinement.

## Code Standards

### Python Best Practices
- Follow PEP 8 style guidelines for Python code
- Use type hints where appropriate (Pydantic models for data validation in `memory/story_state.py`, dataclasses in `settings.py`)
- Write clear, descriptive variable and function names
- Keep functions focused and single-purpose
- Use docstrings for classes and complex functions

### Testing
- Test: `pytest` (runs all tests in `tests/` directory)
- Test with coverage: `pytest --cov`
- Write unit tests for new functionality using pytest
- Place test files in the `tests/` directory with `test_*.py` naming convention

### Development Flow
- Install dependencies: `pip install -r requirements.txt`
- Install dev dependencies: `pip install -r requirements-dev.txt`
- Run the application: `python main.py` (starts web UI on http://localhost:7860)
- Run in CLI mode: `python main.py --cli`

### After Every Change
**IMPORTANT**: After making code changes, always:
1. Run `black .` to format code
2. Run `ruff check .` to lint
3. Run `pytest` to verify tests pass
4. Commit and push changes
5. Verify CI passes on GitHub (check Actions tab)

This ensures code quality and prevents broken builds.

## Repository Structure

- `main.py`: Entry point for the application (supports both web UI and CLI modes)
- `settings.py`: Settings management and model registry
- `settings.json`: User configuration file (not in git, copy from `settings.example.json`)
- `requirements.txt`: Production Python dependencies
- `requirements-dev.txt`: Development and testing dependencies
- `MODELS.md`: Model recommendations and usage guidelines
- `MODELS_ANALYSIS.md`: Detailed model analysis and comparisons
- `agents/`: AI agent implementations
  - `base.py`: Base agent class with common functionality
  - `interviewer.py`: Gathers story requirements from users
  - `architect.py`: Designs world, characters, and plot structure
  - `writer.py`: Generates prose content
  - `editor.py`: Polishes and refines writing
  - `continuity.py`: Detects plot holes and inconsistencies
- `workflows/`: Agent coordination and orchestration
  - `orchestrator.py`: Manages the multi-agent workflow
- `memory/`: Story state management
  - `story_state.py`: Maintains story context across agents
- `utils/`: Utility modules
  - `json_parser.py`: JSON extraction and parsing utilities
  - `logging_config.py`: Logging configuration and setup
- `ui/`: User interface components (NiceGUI)
  - `app.py`: Main NiceGUI application
  - `pages/`: Page components (write, world, settings, models)
  - `components/`: Reusable UI components (header, chat, graph)
- `output/stories/`: Generated story outputs (gitignored)
- `tests/`: Test suite using pytest
- `logs/`: Application logs (logs written to `logs/story_factory.log`)
- `.github/workflows/`: CI/CD workflows

## Key Guidelines

1. **Ollama Integration**: All AI agents use Ollama for local LLM serving. Respect the existing model configuration patterns.

2. **Agent Architecture**: Each agent has a specific role. Maintain separation of concerns:
   - Don't mix agent responsibilities
   - Use the base agent class for common functionality
   - Follow the established agent interface patterns

3. **State Management**: The story state is maintained through the `memory/story_state.py` module. Use it consistently across agents.

4. **Error Handling**: Handle Ollama connection errors and model loading failures gracefully with informative error messages.

5. **Configuration**: Settings are managed through dataclasses in `settings.py`. Use `settings.json` for user configuration.

6. **Dependencies**:
   - Minimize external dependencies
   - When adding new dependencies, add them to `requirements.txt` or `requirements-dev.txt` as appropriate
   - Pin dependency versions with ranges (e.g., `>=4.0.0,<7.0.0`)

7. **Testing**:
   - Write tests for new utility functions and critical logic
   - Mock Ollama API calls in tests to avoid requiring a running Ollama instance
   - Use pytest fixtures for test setup

8. **Documentation**:
   - Update README.md for significant feature changes
   - Update MODELS.md when adding model recommendations
   - Keep docstrings up to date with code changes

9. **Logging**:
   - Logs are written to `logs/story_factory.log`
   - Use Python's `logging` module with `logger = logging.getLogger(__name__)`
   - Follow existing logging patterns for consistency

10. **Web UI**: The NiceGUI interface should remain simple and user-friendly. Test UI changes by running the application.

11. **JSON Parsing**: Use the utilities in `utils/json_parser.py` for extracting and parsing JSON from LLM responses. LLMs may include JSON in markdown code blocks or with surrounding text.

## Workflow Overview

The system follows this workflow:
1. **Interview Phase**: Gather requirements from user
2. **Architecture Phase**: Design world, characters, and plot
3. **Writing Phase**: Iterative chapter generation with Writer → Editor → Continuity checking
4. **Output**: Export completed story as markdown

When modifying the workflow, maintain this structure and ensure agents can communicate effectively through the story state.
