# Architecture Documentation

This document describes the architecture and design patterns used in Story Factory.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Performance Considerations](#performance-considerations)
- [Extending the System](#extending-the-system)

## Overview

Story Factory is a multi-agent AI system for generating creative fiction. It uses a production team metaphor with specialized agents working together in a coordinated workflow.

### Key Design Goals
1. **Modularity**: Each agent is independent and replaceable
2. **Robustness**: Graceful error handling and recovery
3. **Observability**: Comprehensive logging and metrics
4. **Flexibility**: Configurable workflows and interaction modes
5. **Quality**: Iterative refinement and self-critique

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│              (Web UI / CLI / API)                        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator                           │
│  - Workflow coordination                                 │
│  - State management                                      │
│  - Event emission                                        │
│  - Metrics tracking                                      │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Agent Layer       │
    │                     │
    │  ┌──────────────┐   │
    │  │ Interviewer  │   │
    │  └──────────────┘   │
    │         │           │
    │         ▼           │
    │  ┌──────────────┐   │
    │  │  Architect   │   │
    │  └──────────────┘   │
    │         │           │
    │         ▼           │
    │  ┌──────────────┐   │
    │  │   Writer     │───┐
    │  └──────────────┘   │
    │         │           │
    │         ▼           │
    │  ┌──────────────┐   │
    │  │   Editor     │◄──┘
    │  └──────────────┘   │
    │         │           │
    │         ▼           │
    │  ┌──────────────┐   │
    │  │  Continuity  │   │
    │  └──────────────┘   │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │    LLM Backend      │
    │   (Ollama API)      │
    └─────────────────────┘
```

## Core Components

### 1. Agents (`agents/`)

All agents inherit from `BaseAgent` which provides:
- LLM communication with retry logic
- Error handling
- Model and temperature configuration
- Logging integration

#### Agent Responsibilities

**Interviewer** (`interviewer.py`)
- Gathers story requirements through conversation
- Extracts structured brief from dialogue
- Minimal quality requirements (can use faster model)

**Architect** (`architect.py`)
- Designs world, characters, and plot
- Creates chapter outlines
- Needs good reasoning (medium-quality model)

**Writer** (`writer.py`)
- Generates prose for each chapter
- Maintains style and voice
- Requires highest quality model

**Editor** (`editor.py`)
- Polishes and refines prose
- Ensures consistency between chapters
- Medium-quality model sufficient

**Continuity Checker** (`continuity.py`)
- Detects plot holes and inconsistencies
- Validates against outline
- Extracts facts and tracks character arcs
- Low quality acceptable (analytical task)

### 2. Orchestrator (`workflows/orchestrator.py`)

The orchestrator is the central coordinator that:
- Manages workflow state transitions
- Coordinates agent execution
- Emits events for UI updates
- Tracks performance metrics
- Handles saving/loading stories

Key methods:
```python
create_new_story()           # Initialize new story
start_interview()            # Begin interview phase
build_story_structure()      # Architect phase
write_chapter(n)            # Write single chapter with full pipeline
write_all_chapters()        # Generate entire story
save_story() / load_story() # Persistence
```

### 3. Story State (`memory/story_state.py`)

Pydantic models for type-safe state management:

- `StoryBrief`: User requirements (genre, tone, NSFW level, etc.)
- `Character`: Character definitions with arc tracking
- `PlotPoint`: Key plot events with completion tracking
- `Chapter`: Individual chapter with content and status
- `StoryState`: Complete story state with all metadata

All state is serializable to JSON for persistence.

### 4. Settings (`settings.py`)

Centralized configuration with:
- Model registry with metadata (quality, speed, VRAM)
- Per-agent model selection (auto or manual)
- Temperature controls
- Interaction modes
- Validation on load

### 5. Utilities (`utils/`)

**validators.py**: Input validation and sanitization
- Path traversal protection
- Model name validation
- Temperature validation
- Filename sanitization

**json_parser.py**: LLM response parsing
- Extract JSON from markdown code blocks
- Parse to Pydantic models
- Graceful error handling

**logging_config.py**: Structured logging
- Context-aware formatters
- File and console handlers
- Third-party library noise reduction

**metrics.py**: Performance tracking
- Per-operation timing
- Time estimation
- Metrics persistence

## Data Flow

### Story Generation Flow

```
1. Interview Phase
   User ──> Interviewer ──> StoryBrief
   
2. Architecture Phase
   StoryBrief ──> Architect ──> {World, Characters, Plot, Chapters}
   
3. Writing Phase (per chapter)
   ChapterOutline ──> Writer ──> RawContent
                                     │
                                     ▼
   RawContent ──> Editor ──> PolishedContent
                                     │
                                     ▼
   PolishedContent ──> ContinuityChecker ──> Issues?
                                                 │
                    Yes ◄────────────────────────┘
                     │
                     ▼
         Writer (revision with feedback)
         
4. Output
   All Chapters ──> Markdown/Text Export
```

### State Transitions

```
Story Status Flow:
interview → outlining → writing → complete

Chapter Status Flow:
pending → drafting → edited → reviewed → final
```

## Design Patterns

### 1. **Strategy Pattern** - Agent Selection
Different models can be selected per agent based on task requirements:
```python
settings.get_model_for_agent("writer")  # Returns high-quality model
settings.get_model_for_agent("interviewer")  # Returns fast model
```

### 2. **Observer Pattern** - Event Emission
Orchestrator emits events for UI updates without tight coupling:
```python
self._emit("agent_start", "Writer", "Writing chapter...")
self._emit("progress", "Writer", "50% complete")
```

### 3. **Context Manager** - Performance Tracking
Clean resource management for metrics:
```python
with PerformanceTracker(metrics, "write", "Writer", model) as metric:
    content = writer.write_chapter(state, chapter)
    metric.output_length = len(content)
```

### 4. **Retry Pattern** - LLM Resilience
Exponential backoff for transient failures:
```python
for attempt in range(MAX_RETRIES):
    try:
        return client.chat(...)
    except ConnectionError:
        time.sleep(delay)
        delay *= BACKOFF_MULTIPLIER
```

### 5. **Factory Pattern** - Model Auto-Selection
Automatic model selection based on constraints:
```python
def get_model_for_agent(role, vram):
    candidates = filter_by_vram_and_quality(...)
    return optimize_for_role(candidates, role)
```

## Performance Considerations

### Memory Management

1. **Event Pruning**: Events list is capped at MAX_EVENTS (100) to prevent memory leaks
2. **Context Summary**: Story context is compressed for agent prompts
3. **Incremental Saving**: Auto-save after each chapter to prevent data loss

### VRAM Optimization

1. **Per-Agent Models**: Use smaller models for non-critical agents
2. **Auto-Selection**: Automatic model selection based on available VRAM
3. **Sequential Execution**: Agents run sequentially to avoid VRAM contention

### Token Efficiency

1. **Compressed Context**: Only relevant facts and recent events in context
2. **Configurable Max Tokens**: Balance quality vs. generation time
3. **Smart Context Window**: Recent chapters get more detail than older ones

## Extending the System

### Adding a New Agent

1. **Create agent class** inheriting from `BaseAgent`:
```python
class MyAgent(BaseAgent):
    def __init__(self, model=None, settings=None):
        super().__init__(
            name="MyAgent",
            role="My Role",
            system_prompt=MY_PROMPT,
            agent_role="my_agent",
            model=model,
            settings=settings,
        )
    
    def do_task(self, story_state, ...):
        prompt = f"Task: ..."
        response = self.generate(prompt, context=...)
        return parse_response(response)
```

2. **Register in settings.py**:
```python
AGENT_ROLES["my_agent"] = {
    "name": "My Agent",
    "description": "Does something",
    "recommended_quality": 7,
}
```

3. **Integrate in orchestrator**:
```python
self.my_agent = MyAgent(settings=self.settings)

def use_my_agent(self):
    with PerformanceTracker(...) as metric:
        result = self.my_agent.do_task(self.story_state)
```

### Adding a New Interaction Mode

1. **Update settings.py** validation:
```python
valid_modes = [..., "my_mode"]
```

2. **Implement in orchestrator**:
```python
if self.interaction_mode == "my_mode":
    # Custom workflow logic
```

### Adding Export Formats

Implement in orchestrator:
```python
def export_to_epub(self) -> bytes:
    # Generate EPUB from self.story_state
    ...
```

## Configuration Files

- `settings.json`: User configuration (not in git)
- `settings.example.json`: Template for new users
- `.gitignore`: Excludes generated files
- `pytest.ini`: Test configuration
- `requirements.txt`: Python dependencies

## Testing Strategy

1. **Unit Tests**: Each utility module has dedicated tests
2. **Validation Tests**: Settings and inputs are validated
3. **Integration Tests**: (To be added) Test full workflows
4. **Manual Testing**: CLI and UI for end-to-end validation

## Error Handling Philosophy

1. **Fail Fast**: Validate inputs early
2. **Informative Messages**: Errors include actionable guidance
3. **Graceful Degradation**: Non-critical failures don't stop generation
4. **Logging**: All errors logged with context
5. **User-Friendly**: Technical errors translated to user language

## Security Considerations

1. **Path Traversal Prevention**: All file paths validated
2. **Input Sanitization**: User inputs sanitized before use
3. **Model Name Validation**: Prevents injection attacks
4. **No Code Execution**: LLM outputs treated as data only
5. **Local-Only**: No external API calls except to local Ollama

## Future Enhancements

Potential architectural improvements:
- Async/await for parallel agent execution
- Plugin system for custom agents
- API server mode for programmatic access
- Database backend for story persistence
- Distributed execution for multi-GPU setups
- Streaming output for real-time generation
