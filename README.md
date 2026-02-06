# Story Factory

[![CI](https://github.com/Aureliolo/story-factory/actions/workflows/ci.yml/badge.svg)](https://github.com/Aureliolo/story-factory/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Aureliolo/story-factory/branch/main/graph/badge.svg)](https://codecov.io/gh/Aureliolo/story-factory)
[![docstring coverage](docs/badges/interrogate_badge.svg)](https://pypi.org/project/interrogate/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All_Rights_Reserved-red.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Aureliolo/story-factory?utm_source=oss&utm_medium=github&utm_campaign=Aureliolo%2Fstory-factory&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

> **DISCLAIMER**: This is a personal holiday/hobby project created primarily as an experimentation playground for testing AI coding assistants (Claude Code, GitHub Copilot, Cursor, etc.) and exploring local LLM capabilities with Ollama. It's built for my own learning and enjoyment. Feel free to explore, but note that this is not production-grade software and comes with no guarantees. Use at your own risk!

A local AI-powered multi-agent system for generating short stories, novellas, and novels with iterative refinement, self-critique, and plot-hole detection. Everything runs locally on your machine using Ollama.

## Table of Contents

- [Screenshots](#screenshots)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## Screenshots

> **Note:** Screenshots show the application interface as of February 2026. The UI may have been updated since these were captured. The actual interface will reflect the latest features and design improvements.

### Main Interface
![Home Screen](https://github.com/user-attachments/assets/9056d5cb-b696-4510-89ba-3fbfd2fe067e)
*Clean, modern interface with dark mode support and intuitive navigation*

### Project Management
![Projects Page](https://github.com/user-attachments/assets/8b3b165a-96ef-4dd8-940e-ae1917b7c232)
*Manage multiple story projects with backup support*

### Settings & Configuration
![Settings Page](https://github.com/user-attachments/assets/42800304-97b9-43b0-9d05-a77bdedc64a9)
*Fine-tune every aspect: models, temperatures, workflow, and more*

### Model Management
![Models Page](https://github.com/user-attachments/assets/6cf02131-5876-4b1f-9b71-e75adecf7397)
*Download and manage Ollama models with VRAM-aware filtering*

### Analytics Dashboard
![Analytics Page](https://github.com/user-attachments/assets/1c17001c-18a9-4cd3-886f-611293508937)
*Track model performance, quality metrics, and get recommendations*

### Story Templates
![Templates Page](https://github.com/user-attachments/assets/1bfd94aa-fa49-4db5-a985-6e6e924f1a05)
*Built-in genre templates to jumpstart your story creation*

## Features

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| 🤖 **Multi-Agent System** | 5 specialized AI agents working together | ✅ Stable |
| 🌐 **World Building** | Graph-based entity/relationship tracking with visualization | ✅ Stable |
| 🎨 **Modern Web UI** | NiceGUI-powered interface with dark mode | ✅ Stable |
| 📚 **Story Templates** | Pre-built genre templates (Fantasy, Sci-Fi, Romance, etc.) | ✅ Stable |
| 📊 **Analytics Dashboard** | Model performance tracking and recommendations | ✅ Stable |
| 💾 **Multiple Export Formats** | Markdown, Text, HTML, EPUB, PDF | ✅ Stable |
| ⚡ **Background Generation** | Non-blocking UI during story creation | ✅ Stable |
| 🔄 **Version Control** | Chapter history with rollback support | ✅ Stable |
| 🎯 **Adaptive Learning** | Auto-improves based on your preferences | ✅ Stable |
| 🔒 **100% Local & Private** | No cloud, no tracking, complete privacy | ✅ Always |

### AI Agent Production Team

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
    - Chapter regeneration with feedback
    - Version history with rollback
    - Background generation (non-blocking UI)
  - **World Builder Tab**: Visual entity management with graph explorer
    - Import entities from existing text
    - Undo/redo support
  - **Projects Tab**: Manage multiple stories (create, load, duplicate, delete)
  - **Templates Tab**: Pre-built story structures and genre templates
  - **Timeline Tab**: Story event timeline visualization
  - **Comparison Tab**: Side-by-side model comparison for chapter generation
  - **Settings Tab**: Configure models, agent temperatures, and preferences
  - **Models Tab**: Ollama model management with pull/delete functionality
  - **Analytics Tab**: Model performance metrics, quality scores, and recommendations

- **Advanced Features**:
  - **Dark Mode**: Full dark theme support with preference persistence
  - **Keyboard Shortcuts**: Power-user navigation and actions (Ctrl+/)
  - **Iterative Refinement**: Write -> Edit -> Check -> Revise loop for quality
  - **Flexible Output**: Short stories, novellas, or full novels
  - **Multiple Export Formats**: Markdown, Text, HTML, EPUB, PDF
  - **Model Performance Tracking**: Quality scoring, recommendations, and analytics
  - **Local & Private**: Everything runs on your machine

## UI Overview

The application features a comprehensive tabbed interface:

1. **Write Story** - Story creation with two sub-tabs:
   - *Fundamentals*: Interview chat, world overview, story structure
   - *Live Writing*: Chapter-by-chapter writing with version control and regeneration

2. **World Builder** - Entity and relationship management:
   - Visual graph explorer with filters and search
   - Import entities from existing text
   - Undo/redo support

3. **Projects** - Manage multiple story projects

4. **Templates** - Genre templates and story structure presets

5. **Timeline** - Visual story event timeline

6. **Comparison** - Side-by-side model performance testing

7. **Settings** - Model configuration and preferences

8. **Models** - Ollama model management

9. **Analytics** - Model performance metrics and recommendations

## Requirements

### Minimum Requirements
- **Python**: 3.14 or higher
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space for models and outputs
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)

### GPU Requirements (Highly Recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.x or higher
- **For Best Experience**: 24GB VRAM for running multiple larger models

### Software Dependencies
- **Ollama**: Required for local LLM serving ([Installation Guide](https://ollama.com/download))
- **Python Packages**: Automatically installed via `pip install .`

> **Note**: While CPU-only mode is possible, GPU acceleration is strongly recommended for reasonable performance.

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Install Ollama
# Visit https://ollama.com/download and follow instructions for your OS

# 2. Pull a recommended starter model
ollama pull huihui_ai/dolphin3-abliterated:8b

# 3. Clone this repository
git clone https://github.com/Aureliolo/story-factory.git
cd story-factory

# 4. Install Python dependencies
pip install .

# 5. Copy example settings
cp src/settings.example.json src/settings.json

# 6. Launch the web UI
python main.py

# 7. Open your browser to http://localhost:7860
```

That's it! You're ready to create stories. See the [Installation](#installation) section for detailed setup options and multi-model configurations.

## Installation

### 1. Install Ollama

Ollama provides local LLM serving. Choose your platform:

#### Windows
```bash
# Using winget (recommended)
winget install Ollama.Ollama

# Or download installer from https://ollama.com/download
```

#### macOS
```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.com/download
```

#### Linux
```bash
# Quick install script
curl -fsSL https://ollama.com/install.sh | sh

# For manual installation, see https://github.com/ollama/ollama
```

**Verify Installation:**
```bash
ollama --version
# Should output: ollama version is X.X.X
```

### 2. Pull Recommended Models

Start with one of these tested model combinations (updated February 2026):

#### For 8GB VRAM (Entry Level)
```bash
# Best all-rounder for limited VRAM
ollama pull huihui_ai/dolphin3-abliterated:8b
```

#### For 16GB VRAM (Recommended)
```bash
# Balanced quality and speed
ollama pull huihui_ai/dolphin3-abliterated:8b         # Fast, versatile
ollama pull vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0 # Creative writing
```

#### For 24GB VRAM (Premium - February 2026 Update)

```bash
# Best quality multi-agent setup with MoE efficiency
ollama pull huihui_ai/dolphin3-abliterated:8b           # Interviewer
ollama pull vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0   # Writer/Editor
ollama pull huihui_ai/qwen3-abliterated:30b             # Architect/Continuity (MoE: 30B/3B active)
```

> **Note:** Qwen3-30B-A3B is a Mixture-of-Experts (MoE) model that activates only 3B parameters at a time, providing 70B-level reasoning at half the VRAM cost.

See [docs/MODELS.md](docs/MODELS.md) for comprehensive model recommendations and performance comparisons.

### 3. Install Python Dependencies

```bash
cd story-factory
pip install .
```

**Dependencies include:**
- NiceGUI (web interface)
- Ollama Python client
- Pydantic (data validation)
- NetworkX (world graph)
- SQLite (bundled with Python)

### 4. Configure Settings

```bash
# Copy the example settings file
cp src/settings.example.json src/settings.json

# Optional: Edit src/settings.json to customize
# - Ollama URL (if not localhost:11434)
# - Default models
# - Agent temperatures
# - Workflow preferences
```

The application will work with default settings, but you can fine-tune:
- Model selection per agent role
- Temperature (creativity) settings
- Context window sizes
- Workflow modes (checkpoint vs continuous)

### 5. Verify Installation

```bash
# Check that Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON listing your installed models
```

### 6. Launch Application

```bash
# Web UI (recommended)
python main.py

# CLI mode
python main.py --cli

# Custom host/port
python main.py --host 0.0.0.0 --port 8080
```

Open your browser to **http://localhost:7860** (or your custom port).

### Platform-Specific Notes

#### Windows
- Use PowerShell or Windows Terminal
- Ollama runs as a system service
- Check Task Manager if models don't load

#### macOS
- Ollama runs in the background
- Check Activity Monitor for GPU usage
- May need to approve network access

#### Linux
- Ensure CUDA drivers are installed for GPU support
- Check `nvidia-smi` to verify GPU detection
- Ollama service: `sudo systemctl status ollama`

## Usage

### Web UI (Recommended)

The web interface provides the full feature set with an intuitive, modern UI:

```bash
python main.py
```

Open http://localhost:7860 in your browser.

#### Getting Started with Your First Story

1. **Create a Project**: Navigate to the "Projects" tab and click "+ New Project"
2. **Start Writing**: Go to the "Write" tab
3. **Interview Phase**: Chat with the Interviewer agent to define your story
   - Genre and tone
   - Main characters
   - Story length (short story, novella, or novel)
   - Key plot points
4. **Review Architecture**: The Architect will design the world and structure
5. **Generate**: Watch as the Writer, Editor, and Continuity agents collaborate
6. **Export**: Download your finished story in multiple formats

#### UI Tabs Overview

- **Write**: Story creation with interview chat and live chapter writing
- **World**: Visual graph of characters, locations, and relationships
- **Timeline**: Event timeline for your story
- **Compare**: Side-by-side model comparison for testing
- **Projects**: Manage all your story projects
- **Templates**: Genre templates and story structure presets
- **Analytics**: Model performance metrics and quality tracking
- **Settings**: Configure models, temperatures, and workflow
- **Models**: Download and manage Ollama models

#### Keyboard Shortcuts

Press **Ctrl+/** to see all keyboard shortcuts:
- **Ctrl+S**: Save current project
- **Ctrl+N**: New project
- **Ctrl+E**: Export story
- **Ctrl+/**: Show keyboard shortcuts
- **Ctrl+D**: Toggle dark mode

### CLI Mode

For a simpler, terminal-based experience:

```bash
python main.py --cli
```

The CLI mode provides:
- Interactive interview process
- Chapter-by-chapter progress display
- Story generation without browser
- Export to file on completion

#### CLI Examples

```bash
# Start a new story (interactive)
python main.py --cli

# List all saved stories
python main.py --cli --list-stories

# Load and view a saved story
python main.py --cli --load output/stories/my-story.json
```

### Advanced Usage

#### Custom Host and Port

```bash
# Run on all network interfaces
python main.py --host 0.0.0.0 --port 8080

# Development mode with auto-reload
python main.py --reload
```

#### Logging Configuration

```bash
# Enable debug logging
python main.py --log-level DEBUG

# Custom log file
python main.py --log-file /path/to/custom.log

# Disable file logging
python main.py --log-file none
```

#### Configuration via settings.json

Edit `settings.json` to customize:

```json
{
  "default_model": "huihui_ai/dolphin3-abliterated:8b",
  "use_per_agent_models": true,
  "agent_models": {
    "writer": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
    "architect": "huihui_ai/qwen3-abliterated:30b"
  },
  "agent_temperatures": {
    "writer": 0.9,
    "editor": 0.6,
    "continuity": 0.3
  },
  "interaction_mode": "checkpoint",
  "chapters_between_checkpoints": 3
}
```

See [src/settings.example.json](src/settings.example.json) for all available options.

### Export Formats

Story Factory supports multiple export formats:

- **Markdown** (`.md`) - Rich text with formatting
- **Plain Text** (`.txt`) - Simple, portable format
- **HTML** (`.html`) - Web-ready with styling
- **EPUB** (`.epub`) - E-reader compatible
- **PDF** (`.pdf`) - Print-ready document

Export from the web UI via the "Export" button or programmatically via the export service.

## Architecture

```
story-factory/
├── main.py                 # Entry point
├── src/                    # All application source code
│   ├── settings.py         # Settings management & model registry
│   ├── settings.example.json # Example configuration
│   ├── agents/             # AI agent implementations
│   │   ├── base.py         # Base agent class with retry logic
│   │   ├── interviewer.py  # Story requirements gathering
│   │   ├── architect.py    # Structure and character design
│   │   ├── writer.py       # Prose generation
│   │   ├── editor.py       # Refinement and polish
│   │   ├── continuity.py   # Plot hole detection
│   │   └── validator.py    # Response validation
│   ├── memory/             # Data models and persistence
│   │   ├── story_state.py  # Story state (Pydantic models)
│   │   ├── entities.py     # Entity/Relationship models
│   │   ├── world_database.py # SQLite + NetworkX database
│   │   ├── mode_database.py # Model performance database
│   │   ├── mode_models.py  # Performance tracking models
│   │   ├── templates.py    # Template data models
│   │   ├── builtin_templates.py # Built-in story templates
│   │   └── world_quality.py # World quality tracking
│   ├── services/           # Business logic layer
│   │   ├── orchestrator/   # Multi-agent coordination
│   │   │   ├── __init__.py # Main StoryOrchestrator
│   │   │   ├── _interview.py # Interview phase
│   │   │   ├── _structure.py # Architecture phase
│   │   │   ├── _writing.py # Writing phase
│   │   │   ├── _editing.py # Editing phase
│   │   │   └── _persistence.py # State management
│   │   ├── project_service.py # Project CRUD operations
│   │   ├── story_service/  # Story generation workflow
│   │   ├── world_service/  # Entity/world management
│   │   ├── model_service.py # Ollama model operations
│   │   ├── export_service/ # Export to multiple formats
│   │   ├── model_mode_service/ # Model performance tracking
│   │   ├── scoring_service.py # Quality scoring logic
│   │   ├── template_service.py # Story template management
│   │   ├── backup_service.py # Project backup/restore
│   │   ├── import_service.py # Import entities from text
│   │   ├── comparison_service.py # Model comparison testing
│   │   ├── suggestion_service.py # AI-powered suggestions
│   │   ├── world_quality_service/ # World quality enhancement
│   │   ├── timeline_service.py # Timeline management
│   │   ├── calendar_service.py # Calendar view
│   │   ├── conflict_analysis_service.py # Story conflict analysis
│   │   ├── content_guidelines_service.py # Content guidelines
│   │   ├── temporal_validation_service.py # Timeline validation
│   │   ├── world_template_service.py # World templates
│   │   └── llm_client.py   # Unified LLM client
│   ├── ui/                 # NiceGUI web interface
│   │   ├── app.py          # Main application setup
│   │   ├── state.py        # Centralized UI state
│   │   ├── theme.py        # Colors, styles, dark mode
│   │   ├── styles.css      # Custom CSS
│   │   ├── graph_renderer.py # vis.js graph rendering
│   │   ├── keyboard_shortcuts.py # Keyboard shortcut handling
│   │   ├── shortcuts.py    # Shortcut registry
│   │   ├── pages/          # Page components
│   │   └── components/     # Reusable UI components
│   ├── utils/              # Utility modules
│   │   ├── logging_config.py # Logging configuration
│   │   ├── json_parser.py  # JSON extraction from LLM responses
│   │   ├── error_handling.py # Error decorators and handlers
│   │   ├── exceptions.py   # Custom exception hierarchy
│   │   └── ...             # Other utilities
│   └── prompts/            # YAML prompt templates
│       └── templates/      # Template files by agent
├── tests/                  # Test suite (2000+ tests)
│   ├── unit/               # Unit tests
│   ├── component/          # NiceGUI component tests
│   ├── integration/        # Integration tests
│   ├── smoke/              # Quick validation tests
│   └── e2e/                # End-to-end browser tests
├── docs/                   # Documentation
│   ├── codemaps/           # Architecture maps
│   ├── ARCHITECTURE.md     # System design
│   ├── MODELS.md           # Model recommendations
│   ├── CONTRIBUTING.md     # Contribution guidelines
│   ├── TROUBLESHOOTING.md  # Common problems and solutions
│   └── plans/              # Design documents
├── output/                 # Runtime data
│   ├── logs/               # Application logs
│   ├── stories/            # Saved story files
│   ├── worlds/             # World databases
│   └── backups/            # Project backups
└── scripts/                # Developer utilities
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

Settings can be configured via the web UI or by editing `src/settings.json`:

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
pip install -e ".[all]"

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
- **Pytest**: 2000+ tests with 100% coverage on core modules
- **CI/CD**: GitHub Actions with coverage enforcement
- **GitHub Copilot**: Custom instructions configured (see [docs/COPILOT_INSTRUCTIONS.md](docs/COPILOT_INSTRUCTIONS.md))

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_story_service.py

# With coverage report
pytest --cov=. --cov-report=html

# Fast smoke tests only
pytest tests/smoke/

# Integration tests
pytest tests/integration/
```

### Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Troubleshooting

### Ollama Connection Issues

**Problem**: "Ollama offline" error or connection refused

**Solutions**:
1. **Check Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```
2. **Restart Ollama**:
   - Windows: Restart "Ollama" service in Task Manager
   - macOS: `brew services restart ollama`
   - Linux: `sudo systemctl restart ollama`
3. **Verify URL in settings.json**:
   ```json
   {
     "ollama_url": "http://localhost:11434"
   }
   ```
4. **Check firewall**: Ensure port 11434 is not blocked

### Out of Memory / VRAM Issues

**Problem**: Model fails to load or system freezes

**Solutions**:
1. **Use smaller models**: Switch to 8B models instead of 14B+
2. **Reduce context window** in settings:
   ```json
   {
     "context_size": 16384
   }
   ```
3. **Disable per-agent models**: Use single model for all agents
4. **Close other GPU applications**: Free up VRAM
5. **Use more aggressive quantization**: Q4_K_M instead of Q8_0

### Slow Generation Speed

**Problem**: Story generation takes too long

**Solutions**:
1. **Use faster models**: Dolphin 8B is faster than larger models
2. **Reduce max tokens**:
   ```json
   {
     "max_tokens": 4096
   }
   ```
3. **Enable checkpoint mode**: Get feedback every N chapters
4. **Check GPU utilization**: Use `nvidia-smi` to verify GPU usage
5. **Reduce temperature**: Lower values = faster generation

### Chinese Characters in Output

**Problem**: Qwen models output Chinese text

**Solutions**:
1. **Switch to Dolphin**: `huihui_ai/dolphin3-abliterated:8b`
2. **Use Qwen3 v2**: If available, has layer-0 fix
3. **Add system prompt**: Explicitly request English output
4. **Enable validator**: Catches non-English responses

### Installation Issues

**Problem**: pip install fails or missing dependencies

**Solutions**:
1. **Update pip**:
   ```bash
   pip install --upgrade pip
   ```
2. **Use virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install .
   ```
3. **Check Python version**: Must be 3.14+
   ```bash
   python --version
   ```

### UI Not Loading

**Problem**: Browser shows blank page or errors

**Solutions**:
1. **Check console for errors**: Open browser DevTools (F12)
2. **Clear browser cache**: Force refresh with Ctrl+Shift+R
3. **Try different browser**: Chrome, Firefox, or Edge
4. **Check port availability**:
   ```bash
   # Use different port if 7860 is taken
   python main.py --port 8080
   ```
5. **Disable browser extensions**: Ad blockers may interfere

### Story Generation Failures

**Problem**: Story generation stops or produces errors

**Solutions**:
1. **Check model compatibility**: Use recommended models
2. **Increase retry count** in settings:
   ```json
   {
     "llm_max_retries": 5
   }
   ```
3. **Reduce complexity**: Simpler stories are more reliable
4. **Check Ollama logs**: Look for model loading errors
5. **Restart Ollama**: Clear any stuck states

For more help, check:
- **Logs**: `output/logs/story_factory.log`
- **GitHub Issues**: [Report a bug](https://github.com/Aureliolo/story-factory/issues)
- **Model Guide**: [docs/MODELS.md](docs/MODELS.md)

## FAQ

### General Questions

**Q: Do I need an internet connection?**
A: No, Story Factory runs entirely locally. Internet is only needed to download Ollama models initially.

**Q: Is my data private?**
A: Yes, completely. All processing happens on your machine. No data is sent to external servers. All models run locally via Ollama.

**Q: Can I use this commercially?**
A: This software is provided for reference only with all rights reserved. Contact the copyright holder for licensing inquiries. Also check the licenses of the LLM models you use.

**Q: How long does it take to generate a story?**
A: Depends on length and hardware:
- Short story (1 chapter): 5-15 minutes
- Novella (7 chapters): 30-90 minutes
- Novel (20+ chapters): 2-5 hours

Performance varies significantly based on GPU VRAM, model size, and CPU speed.

### Technical Questions

**Q: Can I run this without a GPU?**
A: Yes, but it will be very slow (10-50x slower). GPU with 8GB+ VRAM is strongly recommended for reasonable performance.

**Q: What's the difference between models?**
A: Models vary in:
- **Size** (parameters): 8B, 14B, 30B, 70B, etc.
- **Quality**: Reasoning ability, creativity, instruction-following
- **Speed**: Smaller models are faster but less capable
- **Specialization**: Some excel at creative writing, others at reasoning.

See [docs/MODELS.md](docs/MODELS.md) for detailed comparisons and recommendations.

**Q: Can I use OpenAI/Anthropic models instead of Ollama?**
A: Not currently. The system is designed exclusively for local Ollama models to ensure privacy and offline operation.

**Q: How much disk space do I need?**
A: Plan for:
- Python environment: ~500 MB
- Each 8B model: ~5 GB
- Each 14B model: ~8-10 GB
- Each 30B MoE model: ~15-20 GB
- Generated stories: Variable (typically <100 MB per project)
- Total recommended: 50GB+ free space for comfortable usage

**Q: Can I run multiple stories at once?**
A: The UI supports one active generation at a time to avoid VRAM conflicts. You can manage multiple projects but only generate one story at a time.

**Q: What's the recommended VRAM for different story lengths?**
A:
- **8GB VRAM**: Short stories (1-3 chapters), single model
- **16GB VRAM**: Novellas (5-10 chapters), 2-3 models
- **24GB VRAM**: Full novels (20+ chapters), multi-model setup with MoE models

### Feature Questions

**Q: Can I edit generated text?**
A: Yes! The web UI allows:
- Chapter regeneration with feedback
- Version control with rollback support
- Export and edit in external tools.

**Q: Can I save and resume stories?**
A: Yes, projects are automatically saved and can be resumed at any checkpoint. The system maintains your entire story state including world-building data.

**Q: Can I customize the agents?**
A: Yes, through settings.json you can:
- Assign different models per agent
- Adjust temperature (creativity level)
- Configure context window sizes
- Set workflow preferences (checkpoint vs continuous mode)

Advanced users can modify system prompts in `src/prompts/` (requires code knowledge).

**Q: Does it support languages other than English?**
A: Primarily English. Some models (like Qwen3) support multiple languages, but quality and reliability vary. The system prompts are English-only.

**Q: Can I create my own story templates?**
A: Yes! The Templates tab allows:
- Importing custom templates from JSON
- Creating templates from existing projects
- Sharing templates with others.

**Q: How do I update to the latest version?**

A:

```bash
git pull origin main
pip install --upgrade .
```

Check the release notes for any breaking changes or new requirements.

### Troubleshooting Questions

**Q: Why is my GPU not being used?**
A: Check:
1. CUDA is installed (`nvidia-smi`)
2. Ollama detects GPU (`ollama run <model>` should show GPU info)
3. No other apps are using all VRAM
4. Ollama is using the GPU version (not CPU-only)

**Q: Why do I get "model not found" errors?**
A: The model needs to be pulled first:
```bash
ollama pull <model-name>
```

Check available models: `ollama list`

**Q: Can I use models from HuggingFace?**
A: Yes, but they need to be:
1. Converted to GGUF format
2. Imported to Ollama via Modelfile

See [docs/MODELS.md](docs/MODELS.md) for detailed instructions on importing custom models.

**Q: Why is the UI showing "Offline"?**
A: Ollama isn't running or reachable. Check:
- Ollama is running (`curl http://localhost:11434/api/tags`)
- Correct URL in settings.json
- Firewall isn't blocking port 11434
- Ollama service hasn't crashed

See the [Troubleshooting](#troubleshooting) section for detailed solutions.

## Contributing

Contributions are welcome! This is a hobby project, but I appreciate:

- **Bug Reports**: [Open an issue](https://github.com/Aureliolo/story-factory/issues) with details
- **Feature Requests**: Share your ideas in discussions
- **Code Contributions**: Fork, create a branch, and submit a PR
- **Documentation**: Help improve guides and examples
- **Model Testing**: Share your experiences with different models

### Development Guidelines

1. **Follow existing code style**: Use Ruff for formatting
2. **Write tests**: Maintain 100% coverage on core modules
3. **Update documentation**: Keep README and docs in sync
4. **Test thoroughly**: Run full test suite before submitting PR
5. **Small PRs**: Focus on one feature/fix at a time

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for architectural guidelines.

## Tech Stack

- **[Python](https://www.python.org/)**: 3.14+ - Modern async/await support
- **[NiceGUI](https://nicegui.io/)** - Modern Python web UI framework with real-time updates
- **[Ollama](https://ollama.com/)** - Local LLM serving with efficient model management
- **[NetworkX](https://networkx.org/)** - Graph analysis for world-building relationships
- **[SQLite](https://sqlite.org/)** - Lightweight entity storage with ACID guarantees
- **[vis.js](https://visjs.org/)** - Interactive graph visualization for world explorer
- **[Pydantic](https://pydantic.dev/)** - Runtime data validation and settings management
- **[Pytest](https://pytest.org/)** - Comprehensive testing framework with 2000+ tests
- **[Ruff](https://github.com/astral-sh/ruff)** - Ultra-fast Python linter and formatter

## License

Copyright (c) 2026 Aurelio Amoroso. All Rights Reserved. See [LICENSE](LICENSE) file for details.

## Getting Help

### Documentation

- **[Quick Start Guide](#quick-start)**: Get running in 5 minutes
- **[Installation Guide](#installation)**: Detailed setup instructions
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)**: Solutions to common problems
- **[Model Selection Guide](docs/MODELS.md)**: Choose the best models
- **[Architecture Documentation](docs/ARCHITECTURE.md)**: Understand the system
- **[Contributing Guide](docs/CONTRIBUTING.md)**: Help improve the project

### Support Resources

- **GitHub Issues**: [Report bugs or request features](https://github.com/Aureliolo/story-factory/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/Aureliolo/story-factory/discussions)
- **Logs**: Check `logs/story_factory.log` for detailed error information

### Community

- **LocalLLaMA subreddit**: r/LocalLLaMA for local AI discussions
- **Ollama Community**: [Ollama GitHub](https://github.com/ollama/ollama) and [Discord](https://discord.gg/ollama)
- **NiceGUI Documentation**: [NiceGUI Docs](https://nicegui.io) for UI questions

### Before Asking for Help

1. **Check the logs**: `logs/story_factory.log`
2. **Search existing issues**: Someone may have had the same problem
3. **Read the docs**: Check relevant documentation sections
4. **Verify your setup**:
   - Ollama is running (`curl http://localhost:11434/api/tags`)
   - Models are installed (`ollama list`)
   - Python version is 3.14+ (`python --version`)
   - GPU is detected (`nvidia-smi`)

### Reporting Issues

When reporting an issue, include:
- **Environment**: OS, Python version, Ollama version, GPU
- **Steps to reproduce**: Exact steps that trigger the problem
- **Error messages**: From logs or console
- **Expected vs. actual behavior**: What should happen vs. what does happen
- **Screenshots**: For UI-related issues

## Acknowledgments

This project uses and is inspired by:

- **[Ollama](https://ollama.com)**: Local LLM serving made simple
- **[NiceGUI](https://nicegui.io)**: Beautiful Python web interfaces
- **[NetworkX](https://networkx.org)**: Graph analysis for world-building
- **Eric Hartford**: For creating Dolphin models
- **Qwen Team**: For excellent open-source models
- **LocalLLaMA Community**: For model testing and recommendations

Special thanks to all the model creators, AI researchers, and open-source contributors who make projects like this possible.

## Project Status & Roadmap

### Current Status (v1.0 - February 2026)

✅ **Stable Features**:
- Multi-agent story generation with 5 specialized AI agents
- Modern web UI with NiceGUI 3.x and full dark mode support
- World building system with SQLite + NetworkX graph database
- Story templates and genre presets for quick starts
- Multiple export formats (Markdown, Text, HTML, EPUB, PDF)
- Model performance analytics and quality tracking
- Adaptive learning system that improves based on user preferences
- Background story generation (non-blocking UI)
- Chapter version control with rollback support
- Project backup and restore functionality
- CLI mode for terminal-based story generation

### Recent Updates (2025-2026)

🆕 **Latest Improvements**:
- **MoE Model Support**: Qwen3-30B-A3B for 70B-level reasoning at half the VRAM
- **Enhanced World Quality**: AI-powered entity refinement with early stopping detection
- **Improved Testing**: 2000+ tests with 100% coverage on core modules
- **Performance Optimizations**: LRU caching, incremental graph updates, thread-safe operations
- **Better Error Handling**: Comprehensive exception hierarchy and retry logic
- **Keyboard Shortcuts**: Power-user navigation (`Ctrl+/`)

### Known Limitations

- Single user only (local application)
- One active generation at a time (VRAM constraints)
- Requires good GPU for reasonable performance (CPU mode is very slow)
- English-focused (multilingual support is limited)
- File-based storage (not suitable for large team collaboration)

### Planned Features

📋 **Short-term** (v1.1-v1.2):
- Enhanced world import from existing stories
- Custom agent system prompts via UI configuration
- Story outline variations and alternatives
- Improved error recovery and retry strategies
- Better progress indicators for long-running operations

🔮 **Long-term** (v2.0+):
- Plugin system for custom agents and exporters
- Multi-language story generation support
- Collaborative writing features (multi-user)
- Cloud deployment option for team use
- Advanced A/B testing for model comparison
- Voice narration export
- Integration with external writing tools (Scrivener, etc.)

See [GitHub Issues](https://github.com/Aureliolo/story-factory/issues) for active development and feature requests.

## Purpose

This project serves multiple purposes:

1. **AI Assistant Testing**: Experimenting with different AI coding assistants to see how they handle complex, multi-file Python projects
2. **Local LLM Exploration**: Testing various Ollama models for creative writing tasks
3. **Multi-Agent Architecture**: Learning about coordinating multiple AI agents
4. **Personal Entertainment**: Actually generating stories for fun!

Remember: This is a hobby project. Have fun with it!
