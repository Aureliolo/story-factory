# Progress Tracking UI Mockup

This document shows what the enhanced progress tracking looks like in the UI.

## Before (Old UI)

```
┌────────────────────────────────────────────────────────────┐
│  📖 Generating...                              ⏸  🛑       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━      │
│  (Indeterminate spinner - no clear progress indication)    │
└────────────────────────────────────────────────────────────┘
```

## After (New UI)

```
┌────────────────────────────────────────────────────────────┐
│  Generation Progress                                        │
│                                                             │
│  ✅ Interview → ✅ Architect → 🔵 Writer → ⚪ Editor → ⚪ Continuity │
│     Interview     Architect     Writer    Editor  Continuity│
│                                                             │
│  📖 Writing Chapter 3...                        ⏸  🛑       │
│  Phase: Writer  │  Chapter 3  │  ETA: 2m 30s               │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━      │
│  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 45%  │
└────────────────────────────────────────────────────────────┘
```

## Phase Indicator States

1. **Completed Phase** (✅ green checkmark)
   - Interview complete
   - Architect complete

2. **Current Phase** (🔵 blue icon)
   - Writer (currently active)
   - Shows the actual agent icon

3. **Future Phase** (⚪ grey circle)
   - Editor (not yet started)
   - Continuity (not yet started)

## Progress Bar Behavior

### Interview Phase (10% weight)
```
████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5%
"Processing your story idea..."
```

### Architect Phase (15% weight)
```
██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 18%
"Building world and characters..."
```

### Writer Phase (50% weight - main work)
```
Chapter 1/5 done:
████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 35%
"Writing Chapter 2..."
ETA: 5m 15s

Chapter 3/5 done:
████████████████████████████████░░░░░░░░░░░░░░░ 55%
"Writing Chapter 4..."
ETA: 2m 10s
```

### Editor Phase (15% weight)
```
████████████████████████████████████████░░░░░░░ 78%
"Editing Chapter 3..."
ETA: 45s
```

### Continuity Phase (10% weight)
```
████████████████████████████████████████████░░░ 88%
"Checking Chapter 4 for consistency..."
ETA: 20s
```

### Complete
```
████████████████████████████████████████████████ 100%
"All chapters complete!"
```

## Interactive Elements

1. **Pause Button** (⏸)
   - Pauses generation
   - Changes to ▶ (play) when paused

2. **Cancel Button** (🛑)
   - Cancels generation
   - Disabled after clicking (prevents double-cancel)

3. **Phase Icons**
   - Clickable tooltips showing what each phase does
   - Visual feedback on hover

## ETA Display Format

- Under 60s: "45s"
- 1-60 minutes: "5m 30s"
- Over 1 hour: "1h 15m"

## Real-Time Updates

The UI updates smoothly as WorkflowEvents are emitted:
- Every agent start/complete
- Every revision iteration
- Every phase transition
- ETA recalculates based on actual progress
