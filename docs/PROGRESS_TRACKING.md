# Real-Time Progress Tracking

This feature adds comprehensive real-time progress tracking during story generation with visual indicators, phase tracking, and estimated time remaining (ETA).

## Overview

When generating stories, users now see:

1. **Visual Phase Indicators** - Shows the current phase with icons:
   - 🗨️ Interview (10% weight)
   - 🏛️ Architect (15% weight)
   - ✍️ Writer (50% weight)
   - 📝 Editor (15% weight)
   - ✓ Continuity (10% weight)

2. **Progress Bar** - Determinate progress bar showing overall completion (0-100%)

3. **Current Chapter** - Displays which chapter is currently being processed

4. **ETA** - Estimated time remaining based on:
   - Historical generation times from `mode_database.py`
   - Fallback to time-based estimation if no historical data

## Technical Implementation

### WorkflowEvent Enhancement

The `WorkflowEvent` dataclass now includes:
- `phase: str | None` - Current workflow phase
- `progress: float | None` - Overall progress (0.0-1.0)
- `chapter_number: int | None` - Current chapter being processed
- `eta_seconds: float | None` - Estimated time remaining

### Progress Calculation

Progress is calculated using weighted phases:
```python
phase_weights = {
    "interview": 0.10,    # 10%
    "architect": 0.15,    # 15%
    "writer": 0.50,       # 50% (main work)
    "editor": 0.15,       # 15%
    "continuity": 0.10,   # 10%
}
```

**Important:** For multi-chapter stories, the Writer, Editor, and Continuity phases
cycle for EACH chapter. The actual workflow is:
```
Interview → Architect → [Writer → Editor → Continuity] × N chapters
```

The progress bar accounts for this by distributing each phase's weight across all
chapters. For example, with 3 chapters:
- Interview: 0-10% (once)
- Architect: 10-25% (once)
- Writer Ch1: 25-41.7%, Ch2: 41.7-58.3%, Ch3: 58.3-75%
- Editor Ch1: 75-80%, Ch2: 80-85%, Ch3: 85-90%
- Continuity Ch1: 90-93.3%, Ch2: 93.3-96.7%, Ch3: 96.7-100%

The `_calculate_progress()` method:
1. Determines which phases are complete
2. Adds base progress for completed phases
3. Adds partial progress within current phase
4. For writing phases, calculates chapter-based progress distributed across all chapters

### ETA Calculation

The `_calculate_eta()` method:
1. Queries `mode_database` for historical performance data
2. Uses `avg_tokens_per_second` for the current model and agent role
3. Estimates remaining time based on chapters left
4. Falls back to elapsed time-based estimation if no historical data

### UI Component

`GenerationStatus` component (`ui/components/generation_status.py`) displays:
- Phase indicator row with icons and arrows
- Progress message and phase name
- Current chapter badge
- ETA countdown
- Determinate progress bar
- Pause/Resume and Cancel buttons

The component is updated via `update_from_event(event)` which receives WorkflowEvent objects from the orchestrator.

## Usage

In the Write page, when generating chapters:

1. Click "Write Chapter" or "Write All"
2. The GenerationStatus component appears
3. Watch real-time updates showing:
   - Current phase (with visual indicators)
   - Progress percentage
   - Current chapter
   - Time remaining
4. Phase transitions are animated with icon changes (completed = green checkmark, current = blue, future = grey)

## Testing

Comprehensive test suite in `tests/unit/test_orchestrator_progress.py`:
- 15 tests covering:
  - Progress calculation for all phases
  - ETA estimation with/without historical data
  - Phase tracking and transitions
  - WorkflowEvent field population

Run tests:
```bash
pytest tests/unit/test_orchestrator_progress.py -v
```

## Future Enhancements

Potential improvements:
- More granular progress within each agent (e.g., "Describing character X")
- Historical accuracy tracking for ETA predictions
- Notification when estimated completion time changes significantly
- Progress persistence across browser refreshes
