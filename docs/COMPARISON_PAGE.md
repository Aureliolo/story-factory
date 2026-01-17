# Multi-Model Comparison Page - UI Overview

## Page Layout

### Header Section
```
┌─────────────────────────────────────────────────────────────┐
│  Model Comparison                    [Clear History] (red)  │
└─────────────────────────────────────────────────────────────┘
```

### Model Selection Card
```
┌─────────────────────────────────────────────────────────────┐
│  Select Models to Compare                                   │
│  Choose 2-4 models for side-by-side comparison              │
│                                                              │
│  ┌─────────────────────┬─────────────────────┐             │
│  │ Model 1 (Required)  │ Model 2 (Required)  │             │
│  │ [Dropdown Selector] │ [Dropdown Selector] │             │
│  └─────────────────────┴─────────────────────┘             │
│  ┌─────────────────────┬─────────────────────┐             │
│  │ Model 3 (Optional)  │ Model 4 (Optional)  │             │
│  │ [Dropdown Selector] │ [Dropdown Selector] │             │
│  └─────────────────────┴─────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### Chapter Selection Card
```
┌─────────────────────────────────────────────────────────────┐
│  Chapter Selection                                          │
│                                                              │
│  [Chapter Dropdown: Chapter 1: Title]  [Generate Compare ➜] │
└─────────────────────────────────────────────────────────────┘
```

### Progress Indicator
```
Progress: ✓ Celeste V1.9 12B complete (100%)
```

### Comparison Results (Side-by-Side)
```
┌───────────────────────────┬───────────────────────────┐
│ ✓ Celeste V1.9 12B       │   Dark Champion 18B MOE   │
├───────────────────────────┼───────────────────────────┤
│ Words: 2,487             │ Words: 2,312             │
│ Time: 145.2s             │ Time: 132.8s             │
│ Speed: 1,027 w/m         │ Speed: 1,045 w/m         │
├───────────────────────────┼───────────────────────────┤
│ ▼ Preview                │ ▼ Preview                │
│   [Chapter content...]   │   [Chapter content...]   │
│                          │                          │
│ (Selected - Green border)│ [Select This Version]    │
└───────────────────────────┴───────────────────────────┘
```

### Selection Dialog
```
┌─────────────────────────────────────┐
│ Why did you choose this model?      │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Optional notes                  │ │
│ │ e.g., Better dialogue, more     │ │
│ │ engaging prose...               │ │
│ └─────────────────────────────────┘ │
│                                     │
│         [Cancel]  [Confirm Selection]│
└─────────────────────────────────────┘
```

### Comparison History
```
┌─────────────────────────────────────────────────────────────┐
│  Recent Comparisons                                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 🕐 Chapter 3 - 2026-01-17 17:49                        │ │
│  │                      Winner: Celeste V1.9 12B   [View] │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 🕐 Chapter 2 - 2026-01-17 17:30                        │ │
│  │                 Winner: Dark Champion 18B MOE   [View] │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Features Implemented

### 1. Model Selection
- Dropdown selectors for 2-4 models
- Shows installed models only
- Displays model names with full IDs in parentheses
- Pre-populates with `comparison_models` from settings
- Validates minimum 2 models, maximum 4 models
- Prevents duplicate model selection

### 2. Chapter Selection
- Dropdown of available chapters from current project
- Shows chapter number and title
- Disabled if no chapters exist (prompts user to build structure first)

### 3. Generation Process
- Real-time progress updates during generation
- Shows current model being generated
- Displays agent name and activity
- Progress percentage (0-100%)
- Sequential generation (one model at a time)

### 4. Comparison Display
- Grid layout (2 columns for readability)
- Result cards for each model:
  - Model name (extracted and cleaned)
  - Metrics: word count, generation time, words/minute
  - Content preview (expandable, max 2000 chars shown initially)
  - Selection button (if not already selected)
- Green border on selected version
- Red border on failed generations
- Error messages displayed when generation fails

### 5. Selection Mechanism
- Click "Select This Version" on any result
- Dialog prompts for optional notes
- Records selection with timestamp
- Updates display to highlight selected version
- Stores selection for analytics

### 6. Comparison History
- Shows last 5 comparisons
- Displays chapter number, timestamp, and winner
- "View" button to load previous comparison
- Smooth scroll to results when viewing history

### 7. Analytics Support
- `ComparisonRecord` stores all comparison data
- Tracks model performance metrics
- Win rate calculation per model
- User notes for qualitative feedback
- History persistence in service (in-memory)

## Navigation
- Added to main navigation bar as "Compare" with compare_arrows icon
- Route: `/compare`
- Accessible from any page

## Error Handling
- No project loaded: Shows friendly message to create/load project
- No chapters available: Prompts to build story structure
- Validation errors: Clear user notifications
- Generation errors: Captured and displayed per model
- Failed generations don't block other models

## Future Enhancements (Not Implemented)
- Diff highlighting (word-level differences)
- Persistence of comparison history to disk
- Export comparison results
- A/B testing mode with blind comparison
- Statistical analysis of model performance
- Parallel generation (requires thread-safe orchestrators)
