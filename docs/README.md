# Story Factory Documentation

This directory contains comprehensive documentation for the Story Factory project.

## Core Documentation

### [ARCHITECTURE.md](ARCHITECTURE.md)
System architecture, directory structure, design patterns, and API patterns. Essential reference for understanding the codebase structure.

### [MODELS.md](MODELS.md)
Comprehensive guide to LLM model selection for Story Factory's multi-agent system. Includes:
- Model recommendations for each agent role
- Quantization strategies for different VRAM configurations
- Performance benchmarks and research findings
- HuggingFace model installation guides

### [CODE_QUALITY.md](CODE_QUALITY.md)
Tracks code quality improvements and fixes. Documents resolved issues in:
- Security (SQL injection prevention, XSS escaping)
- Reliability (thread safety, connection leak prevention)
- Performance (LRU caching, incremental updates)

### [TEST_QUALITY.md](TEST_QUALITY.md)
Testing standards and coverage report:
- 849 tests with 100% coverage on core modules
- Test organization and patterns
- CI configuration and best practices

## Feature Documentation

### [UX_UI_IMPROVEMENTS.md](UX_UI_IMPROVEMENTS.md)
UI/UX enhancements including:
- Dark mode implementation
- Reusable component library
- Keyboard shortcuts
- Accessibility improvements

### [UNDO_REDO.md](UNDO_REDO.md)
Undo/redo functionality using command pattern:
- Keyboard shortcuts (Ctrl+Z, Ctrl+Y)
- Supported actions across World and Settings pages
- API reference and implementation details

### [TEMPLATES.md](TEMPLATES.md)
Story template system documentation:
- Built-in genre templates (Mystery, Romance, Sci-Fi, Fantasy, Thriller)
- Structure presets (Three-Act, Hero's Journey, Save the Cat)
- Custom template creation and sharing

## Development Guides

### [COPILOT_INSTRUCTIONS.md](COPILOT_INSTRUCTIONS.md)
Guide to GitHub Copilot custom instructions for this repository:
- Repository-wide instructions
- Path-specific instructions for tests, UI pages, and agents
- Best practices for maintaining code quality

## Future Plans

### [plans/](plans/)
Design documents for planned features:

- **[2026-01-15-model-modes-scoring-design.md](plans/2026-01-15-model-modes-scoring-design.md)** - Model orchestration system with generation modes, scoring, and adaptive learning
- **[2026-01-15-nicegui-testing-strategy.md](plans/2026-01-15-nicegui-testing-strategy.md)** - NiceGUI testing framework integration plan

## Project Root Documentation

- **[../README.md](../README.md)** - Main project README with installation, usage, and feature overview
- **[../CLAUDE.md](../CLAUDE.md)** - Claude Code assistant guidance (coding standards, commands, architecture)

## Documentation Updates

Last updated: January 2026

### Recent Changes
- Removed obsolete implementation-complete documentation (UI mockups, feature specs)
- Consolidated feature documentation into README and feature-specific docs
- Organized documentation into clear categories
- Added this documentation index

### Documentation Standards
- Keep docs up-to-date with implementation
- Remove docs when features are complete and integrated into main README
- Use docs/ for technical references and guides, not temporary planning docs
- Planning docs go in docs/plans/ with date prefixes
