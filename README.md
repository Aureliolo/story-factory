# Story Factory

A local AI-powered multi-agent system for generating short stories, novellas, and novels with iterative refinement, self-critique, and plot-hole detection.

## Features

- **Multi-Agent Production Team**: 5 specialized AI agents working together
  - **Interviewer**: Gathers story requirements through conversation
  - **Architect**: Designs world, characters, and plot structure
  - **Writer**: Creates prose with genre-appropriate style
  - **Editor**: Polishes and refines the writing
  - **Continuity Checker**: Detects plot holes and inconsistencies

- **Iterative Refinement**: Write → Edit → Check → Revise loop for quality
- **NSFW Support**: Uncensored models for adult content
- **Flexible Output**: Short stories, novellas, or full novels
- **User Interaction**: Configurable checkpoints for feedback and steering
- **Local & Private**: Everything runs on your machine

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (24GB recommended for 70B models)
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
ollama pull tohur/natsumura-storytelling-rp-llama-3.1

# For better quality (requires 20GB+ VRAM)
ollama pull vanilj/midnight-miqu-70b-v1.5:Q4_K_M
```

See [MODELS.md](MODELS.md) for full model recommendations.

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

## Configuration

Settings can be configured via the web UI or by editing `settings.json`:

```json
{
  "default_model": "huihui_ai/qwen3-abliterated:32b",
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

Copy `settings.example.json` to `settings.json` to get started.

## Project Structure

```
story-factory/
├── main.py                 # Entry point
├── settings.py             # Settings management & model registry
├── settings.json           # User settings (not in git)
├── settings.example.json   # Example settings template
├── requirements.txt        # Python dependencies
├── agents/                 # AI agent implementations
│   ├── base.py             # Base agent class
│   ├── interviewer.py      # Story requirements gathering
│   ├── architect.py        # Structure and character design
│   ├── writer.py           # Prose generation
│   ├── editor.py           # Refinement and polish
│   └── continuity.py       # Plot hole detection
├── workflows/
│   └── orchestrator.py     # Agent coordination
├── memory/
│   └── story_state.py      # Story state management
├── utils/
│   └── json_parser.py      # JSON extraction utilities
├── ui/
│   └── gradio_app.py       # Web interface
└── output/
    └── stories/            # Generated stories
```

## Workflow

```
User Input → Interviewer → Architect → [Writer → Editor → Continuity] × N → Final Story
                                              ↑_______revision loop_______↓
```

1. **Interview Phase**: The Interviewer asks about your story idea, genre, tone, characters, and content preferences
2. **Architecture Phase**: The Architect creates world-building, character profiles, plot outline, and chapter structure
3. **Writing Phase**: For each chapter:
   - Writer drafts the content
   - Editor polishes the prose
   - Continuity Checker validates consistency
   - If issues found, loop back to Writer (max 3 iterations)
4. **Output**: Complete story exported as markdown

## Interaction Modes

| Mode | Description |
|------|-------------|
| **Minimal** | Only asks at start, shows final result |
| **Checkpoint** | Reviews every N chapters |
| **Interactive** | Reviews each chapter |
| **Collaborative** | Frequent interaction, steer mid-scene |

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM serving
- [Gradio](https://gradio.app/) - Web UI framework
- [CrewAI](https://crewai.com/) - Multi-agent inspiration
