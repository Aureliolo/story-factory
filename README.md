# Story Factory

[![CI](https://github.com/Aureliolo/story-factory/actions/workflows/ci.yml/badge.svg)](https://github.com/Aureliolo/story-factory/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **DISCLAIMER**: This is a personal holiday/hobby project created primarily as an experimentation playground for testing AI coding assistants (Claude Code, GitHub Copilot, Cursor, etc.) and exploring local LLM capabilities with Ollama. It's built for my own learning and enjoyment. Feel free to explore, but note that this is not production-grade software and comes with no guarantees. Use at your own risk!

A local AI-powered multi-agent system for generating short stories, novellas, and novels with iterative refinement, self-critique, and plot-hole detection. Everything runs locally on your machine using Ollama.

## Features

- **Multi-Agent Production Team**: 5 specialized AI agents working together
  - **Interviewer**: Gathers story requirements through conversation
  - **Architect**: Designs world, characters, and plot structure
  - **Writer**: Creates prose with genre-appropriate style
  - **Editor**: Polishes and refines the writing
  - **Continuity Checker**: Detects plot holes and inconsistencies

- **World Building System**: SQLite + NetworkX powered entity/relationship database
  - Characters, locations, items, factions tracking
  - Interactive graph visualization with vis.js
  - Relationship mapping and path finding
  - Community/cluster detection

- **Modern Web UI**: Built with NiceGUI
  - **Write Story Tab**: Interview-based story creation with live writing
  - **World Builder Tab**: Visual entity management with graph explorer
  - **Projects Tab**: Manage multiple stories
  - **Settings Tab**: Configure models and preferences
  - **Models Tab**: Ollama model management with VRAM detection

- **Iterative Refinement**: Write -> Edit -> Check -> Revise loop for quality
- **Flexible Output**: Short stories, novellas, or full novels
- **Multiple Export Formats**: Markdown, Text, HTML, EPUB, PDF
- **Local & Private**: Everything runs on your machine

## Screenshots

The UI is organized into tabs:

1. **Write Story** - Two sub-tabs:
   - *Fundamentals*: Interview chat, world overview, story structure
   - *Live Writing*: Chapter-by-chapter writing with real-time feedback

2. **World Builder** - Entity management:
   - Left panel: Entity browser with type filters and search
   - Center: Interactive vis.js graph visualization
   - Right panel: Entity editor with attributes

3. **Projects** - Project management with create, load, duplicate, delete

4. **Settings** - Model configuration, agent temperatures, preferences

5. **Models** - Ollama model management with pull/delete functionality

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (24GB recommended for larger models)
- **CUDA**: 11.x or higher
- **Python**: 3.13+
- **Ollama**: For local LLM serving

## Installation

### 1. Install Ollama

```bash
# Windows
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a Model

```bash
# Recommended starter model (8B, fits on most GPUs)
ollama pull huihui_ai/qwen3-abliterated:8b

# For better quality (requires 16GB+ VRAM)
ollama pull huihui_ai/qwen3-abliterated:14b
```

See [docs/MODELS.md](docs/MODELS.md) for full model recommendations.

### 3. Install Python Dependencies

```bash
cd story-factory
pip install -r requirements.txt
```

## Usage

### Web UI (Recommended)

```bash
python main.py
```

Open http://localhost:7860 in your browser.

### CLI Mode

```bash
python main.py --cli
```

## Architecture

```
story-factory/
├── main.py                 # Entry point
├── settings.py             # Settings management & model registry
├── agents/                 # AI agent implementations
│   ├── base.py             # Base agent class
│   ├── interviewer.py      # Story requirements gathering
│   ├── architect.py        # Structure and character design
│   ├── writer.py           # Prose generation
│   ├── editor.py           # Refinement and polish
│   ├── continuity.py       # Plot hole detection
│   └── validator.py        # Response validation
├── workflows/
│   └── orchestrator.py     # Agent coordination
├── memory/
│   ├── story_state.py      # Story state management
│   ├── entities.py         # Entity/Relationship models
│   └── world_database.py   # SQLite + NetworkX database
├── services/               # Business logic layer
│   ├── project_service.py  # Project CRUD
│   ├── story_service.py    # Story generation
│   ├── world_service.py    # Entity management
│   ├── model_service.py    # Ollama operations
│   └── export_service.py   # Export formats
├── ui/                     # NiceGUI web interface
│   ├── app.py              # Main application
│   ├── state.py            # Centralized UI state
│   ├── theme.py            # Colors, styles
│   ├── styles.css          # Custom CSS
│   ├── graph_renderer.py   # vis.js graph rendering
│   ├── keyboard_shortcuts.py # Keyboard shortcut handling
│   ├── pages/              # Page components
│   │   ├── write.py        # Write Story page
│   │   ├── world.py        # World Builder page
│   │   ├── projects.py     # Projects page
│   │   ├── settings.py     # Settings page
│   │   └── models.py       # Models page
│   └── components/         # Reusable UI components
│       ├── header.py       # App header with project selector
│       ├── chat.py         # Chat interface
│       ├── graph.py        # Graph component wrapper
│       ├── entity_card.py  # Entity display
│       └── common.py       # Shared components (loading, dialogs, etc.)
└── tests/                  # Test suite
    ├── unit/               # Unit tests
    ├── component/          # NiceGUI component tests
    ├── integration/        # Integration tests
    └── e2e/                # End-to-end tests
```

## Workflow

```
User Input -> Interviewer -> Architect -> [Writer -> Editor -> Continuity] x N -> Final Story
                                              ^_______revision loop_______v
```

1. **Interview Phase**: The Interviewer asks about your story idea, genre, tone, characters
2. **Architecture Phase**: The Architect creates world-building, character profiles, plot outline
3. **Writing Phase**: For each chapter:
   - Writer drafts the content
   - Editor polishes the prose
   - Continuity Checker validates consistency
   - If issues found, loop back to Writer (max 3 iterations)
4. **Output**: Complete story exported in your preferred format

## Configuration

Settings can be configured via the web UI or by editing `settings.json`:

```json
{
  "default_model": "huihui_ai/qwen3-abliterated:14b",
  "context_size": 32768,
  "interaction_mode": "checkpoint",
  "chapters_between_checkpoints": 3,
  "use_per_agent_models": true,
  "agent_temperatures": {
    "writer": 0.9,
    "editor": 0.6
  }
}
```

## Development

### Quick Start

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term

# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy .
```

### Code Quality

- **Ruff**: Fast Python formatter and linter
- **MyPy**: Type checking
- **Pytest**: Testing framework with coverage

## Tech Stack

- **[NiceGUI](https://nicegui.io/)**: Modern Python web UI framework
- **[Ollama](https://ollama.com/)**: Local LLM serving
- **[NetworkX](https://networkx.org/)**: Graph analysis for world-building
- **[SQLite](https://sqlite.org/)**: Entity storage
- **[vis.js](https://visjs.org/)**: Interactive graph visualization
- **[Pydantic](https://pydantic.dev/)**: Data validation

## License

MIT License - See LICENSE file for details.

## Purpose

This project serves multiple purposes:

1. **AI Assistant Testing**: Experimenting with different AI coding assistants to see how they handle complex, multi-file Python projects
2. **Local LLM Exploration**: Testing various Ollama models for creative writing tasks
3. **Multi-Agent Architecture**: Learning about coordinating multiple AI agents
4. **Personal Entertainment**: Actually generating stories for fun!

Remember: This is a hobby project. Have fun with it!
