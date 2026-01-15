# Design: Model Modes & Adaptive Scoring System

**Date:** 2026-01-15
**Status:** Design Complete, Pending Implementation
**Hardware Reference:** RTX 4090 (24GB VRAM)

---

## Overview

A flexible model orchestration system with three core capabilities:

1. **Generation Modes** - Presets and custom model combinations per agent role
2. **Scoring System** - Comprehensive quality tracking (LLM-judged, implicit signals, user ratings)
3. **Adaptive Learning** - Self-tuning recommendations based on accumulated scores

---

## Goals

- **Discovery/Experimentation**: Try different model combos to learn what works best
- **Quality Optimization**: Use the best models, swapping as needed for VRAM
- **Speed vs Quality Tradeoff**: Let users choose fast-draft vs high-quality modes
- **Persistent Learning**: Track model performance across projects

---

## Architecture

### New Service: `ModelModeService`

```
┌─────────────────────────────────────────────────────────┐
│                    ModelModeService                      │
├─────────────────────────────────────────────────────────┤
│  Modes           │  Scoring          │  Learning        │
│  ─────           │  ───────          │  ────────        │
│  - Presets       │  - SQLite DB      │  - Tuning LLM    │
│  - Custom        │  - Per-agent      │  - Auto/Manual   │
│  - VRAM mgmt     │  - Per-chapter    │  - Suggestions   │
└─────────────────────────────────────────────────────────┘
```

**Location:** `services/model_mode_service.py`

**Responsibilities:**
- Manage mode definitions (presets + custom)
- VRAM-aware model loading/unloading
- Score collection and persistence
- Learning engine (LLM-based tuning)
- Recommendation generation

---

## Component 1: Generation Modes

### Mode Data Model

```python
@dataclass
class GenerationMode:
    id: str                          # Unique identifier
    name: str                        # Display name
    description: str                 # User-facing description
    agent_models: dict[str, str]     # agent_role → model_id
    agent_temperatures: dict[str, float]
    vram_strategy: str               # "sequential", "parallel", "adaptive"
    is_preset: bool                  # Built-in vs user-created
    is_experimental: bool            # For learning mode (tries variations)
```

### VRAM Strategies

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `sequential` | Fully unload between agents, max VRAM per model | 70B models |
| `parallel` | Keep multiple small models loaded | Fast 8B combos |
| `adaptive` | Keep validator if space, swap otherwise | Balanced (default) |

### Built-in Presets

#### 1. `quality_max` - Maximum Quality (Sequential)

| Agent | Model | VRAM |
|-------|-------|------|
| Architect | `huihui_ai/qwen3-abliterated:30b` | 18GB |
| Writer | `vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0` | 14GB |
| Editor | Same as Writer | - |
| Continuity | `DeepSeek-R1-Distill-Qwen-14B` | 10GB |
| Interviewer | `huihui_ai/dolphin3-abliterated:8b` | 5GB |
| Validator | `SmolLM2-1.7B-Instruct` | 1.2GB |

**VRAM Strategy:** `sequential` (unload between agents)
**Use Case:** Best possible output quality, slower generation

#### 2. `quality_creative` - Creative Focus

| Agent | Model | VRAM |
|-------|-------|------|
| Architect | `huihui_ai/qwen3-abliterated:30b` | 18GB |
| Writer | `TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit` | 14GB |
| Editor | `vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0` | 14GB |
| Continuity | `huihui_ai/dolphin3-abliterated:8b` | 5GB |
| Interviewer | `huihui_ai/dolphin3-abliterated:8b` | 5GB |
| Validator | `SmolLM2-1.7B-Instruct` | 1.2GB |

**VRAM Strategy:** `sequential`
**Use Case:** Maximum prose creativity (Dark Champion + Celeste)

#### 3. `balanced` - Balanced Quality/Speed

| Agent | Model | VRAM |
|-------|-------|------|
| Architect | `qwen3:14b` | 10GB |
| Writer | `vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0` | 14GB |
| Editor | Same as Writer | - |
| Continuity | `huihui_ai/qwen3-abliterated:8b` | 5GB |
| Interviewer | `huihui_ai/dolphin3-abliterated:8b` | 5GB |
| Validator | `qwen3:0.6b` | 0.5GB |

**VRAM Strategy:** `adaptive`
**Use Case:** Good quality with reasonable speed

#### 4. `draft_fast` - Fast Drafting

| Agent | Model | VRAM |
|-------|-------|------|
| Architect | `huihui_ai/qwen3-abliterated:8b` | 5GB |
| Writer | `huihui_ai/dolphin3-abliterated:8b` | 5GB |
| Editor | Same as Writer | - |
| Continuity | `qwen3:4b` | 3GB |
| Interviewer | `huihui_ai/dolphin3-abliterated:8b` | 5GB |
| Validator | `qwen3:0.6b` | 0.5GB |

**VRAM Strategy:** `parallel` (multiple models loaded)
**Use Case:** Rapid iteration, getting ideas down fast

#### 5. `experimental` - Learning Mode

**Special mode that varies model assignments to gather comparative data.**

- Randomly varies models within quality tiers
- Tracks A/B comparisons (same chapter, different models)
- Builds the scoring dataset fastest
- Auto-adjusts based on emerging patterns

**VRAM Strategy:** `adaptive`
**Use Case:** Discovering optimal model combinations

---

## Component 2: Scoring System

### Database Schema

**File:** `output/model_scores.db`

```sql
-- Per-generation scores (granular tracking)
CREATE TABLE generation_scores (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Context
    project_id TEXT NOT NULL,
    chapter_id TEXT,
    agent_role TEXT NOT NULL,
    model_id TEXT NOT NULL,
    mode_name TEXT NOT NULL,
    genre TEXT,

    -- Performance metrics
    tokens_generated INTEGER,
    time_seconds FLOAT,
    tokens_per_second FLOAT,
    vram_used_gb FLOAT,

    -- Quality scores (0-10 scale)
    prose_quality FLOAT,            -- LLM-judged
    instruction_following FLOAT,    -- LLM-judged against brief
    consistency_score FLOAT,        -- Derived from continuity issues

    -- Implicit signals
    was_regenerated BOOLEAN DEFAULT FALSE,
    edit_distance INTEGER,          -- Levenshtein from original
    user_rating INTEGER,            -- Optional 1-5 stars

    -- For A/B comparisons
    prompt_hash TEXT,               -- Hash of input prompt

    -- Indexes
    INDEX idx_model (model_id),
    INDEX idx_role (agent_role),
    INDEX idx_genre (genre),
    INDEX idx_project (project_id)
);

-- Aggregated model performance (materialized view)
CREATE TABLE model_performance (
    model_id TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    genre TEXT,                     -- NULL = overall

    avg_prose_quality FLOAT,
    avg_instruction_following FLOAT,
    avg_consistency FLOAT,
    avg_tokens_per_second FLOAT,

    sample_count INTEGER,
    last_updated DATETIME,

    PRIMARY KEY (model_id, agent_role, genre)
);

-- Tuning recommendations history
CREATE TABLE recommendations (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    recommendation_type TEXT,       -- "model_swap", "temp_adjust", etc.
    current_value TEXT,
    suggested_value TEXT,
    reason TEXT,
    confidence FLOAT,
    evidence_json TEXT,

    -- Outcome tracking
    was_applied BOOLEAN,
    user_feedback TEXT              -- "accepted", "rejected", "ignored"
);
```

### Scoring Sources

| Metric | Source | When Collected |
|--------|--------|----------------|
| `prose_quality` | LLM Judge (SmolLM2-1.7B) | After each chapter |
| `instruction_following` | LLM Judge checks against StoryBrief | After each chapter |
| `consistency_score` | Derived from ContinuityChecker issues | After continuity pass |
| `tokens_per_second` | Measured during generation | Real-time |
| `was_regenerated` | Track "Regenerate" button clicks | User action |
| `edit_distance` | Compare pre/post human edits | When user saves edits |
| `user_rating` | Optional 1-5 star rating UI | User-triggered |

### LLM Judge Prompt

```
You are evaluating the quality of AI-generated story content.

**Story Brief:**
Genre: {genre}
Tone: {tone}
Themes: {themes}
Target audience: {audience}

**Content to evaluate:**
{chapter_content}

Rate each dimension from 0-10:

1. PROSE_QUALITY: Creativity, flow, engagement, vocabulary variety
2. INSTRUCTION_FOLLOWING: Adherence to genre, tone, themes, constraints

Respond in JSON:
{"prose_quality": X.X, "instruction_following": X.X, "brief_notes": "..."}
```

### Implicit Signal Collection

```python
class ScoreCollector:
    """Collects implicit quality signals during workflow."""

    def on_regenerate(self, chapter_id: str, agent_role: str):
        """User clicked regenerate - negative signal."""
        self._record_implicit("regenerate", chapter_id, agent_role)

    def on_edit(self, chapter_id: str, original: str, edited: str):
        """User manually edited content."""
        distance = levenshtein_distance(original, edited)
        self._record_implicit("edit", chapter_id, edit_distance=distance)

    def on_accept(self, chapter_id: str):
        """User accepted content without changes - positive signal."""
        self._record_implicit("accept", chapter_id)
```

---

## Component 3: Adaptive Learning System

### Tuning LLM

**Model:** SmolLM2-1.7B-Instruct or Qwen3-4B (smallest capable)

**Runs:** Asynchronously after generation (non-blocking)

**Input:** Accumulated scores from `generation_scores` table

**Output:** `TuningRecommendation` objects

### Recommendation Data Model

```python
@dataclass
class TuningRecommendation:
    id: str
    timestamp: datetime
    recommendation_type: str    # "model_swap", "temp_adjust", "mode_change"
    current_value: str
    suggested_value: str
    reason: str                 # Human-readable explanation
    confidence: float           # 0-1, based on sample size and variance
    evidence: dict              # Supporting statistics

    # For model swaps
    affected_role: str | None
    expected_improvement: float | None
```

### Learning Triggers

| Mode | Behavior |
|------|----------|
| `off` | No automatic analysis |
| `after_project` | Analyze after story completion |
| `periodic` | Every N chapters (default: 5) |
| `continuous` | Background analysis, surface when confident |

### Autonomy Levels

| Level | Auto-Apply | Notify | Best For |
|-------|------------|--------|----------|
| `manual` | Never | Always ask | Production projects |
| `cautious` | Temp adjustments only | Model swaps | Careful experimentation |
| `balanced` | When confidence > 80% | Otherwise prompt | General use |
| `aggressive` | All recommendations | Just notify | Rapid learning |
| `experimental` | Everything + variations | Summary only | Data gathering |

### Tuning Constraints

1. **Minimum samples:** 5 generations before making recommendations
2. **Installed only:** Won't suggest models not in Ollama
3. **VRAM aware:** Respects current VRAM limits
4. **Confidence threshold:** Varies by autonomy level
5. **Cooldown:** Won't re-suggest rejected recommendations for N chapters

### Tuning LLM Prompt

```
You are analyzing LLM model performance data for a story generation system.

**Current Mode:** {mode_name}
**Agent Models:**
{agent_models_table}

**Performance Data (last {n} generations):**
{performance_summary}

**Available Models:**
{available_models}

Based on this data, suggest improvements. Consider:
- Quality scores by model and role
- Speed/quality tradeoffs
- Genre-specific performance
- VRAM constraints: {vram_gb}GB available

Respond in JSON:
{
  "recommendations": [
    {
      "type": "model_swap",
      "role": "writer",
      "current": "model_a",
      "suggested": "model_b",
      "reason": "...",
      "confidence": 0.85,
      "expected_improvement": "+12% prose quality"
    }
  ],
  "analysis_notes": "..."
}
```

---

## UI Integration

### Settings Page Additions

**New Section: "Generation Mode"**

```
┌─────────────────────────────────────────────────────────┐
│ Generation Mode                                          │
├─────────────────────────────────────────────────────────┤
│ Mode: [▼ Quality - Creative     ]  [Customize...]       │
│                                                          │
│ Learning: [▼ Balanced           ]                        │
│ ☑ After each project                                    │
│ ☑ Every 5 chapters                                      │
│ ☐ Continuous (experimental)                             │
│                                                          │
│ [View Model Performance →]                              │
└─────────────────────────────────────────────────────────┘
```

**Customize Mode Dialog:**

```
┌─────────────────────────────────────────────────────────┐
│ Custom Mode Configuration                                │
├─────────────────────────────────────────────────────────┤
│ Name: [My Custom Mode          ]                        │
│                                                          │
│ Agent          Model                      Temp          │
│ ─────────────────────────────────────────────────       │
│ Architect      [▼ Qwen3-30B-A3B        ]  [0.3]        │
│ Writer         [▼ Celeste V1.9 12B     ]  [0.9]        │
│ Editor         [▼ Same as Writer       ]  [0.6]        │
│ Continuity     [▼ DeepSeek-R1-14B      ]  [0.2]        │
│ Interviewer    [▼ Dolphin 3.0 8B       ]  [0.7]        │
│ Validator      [▼ SmolLM2-1.7B         ]  [0.1]        │
│                                                          │
│ VRAM Strategy: [▼ Adaptive             ]                │
│                                                          │
│ [Cancel]                            [Save as Preset]    │
└─────────────────────────────────────────────────────────┘
```

### New Analytics Page (`/analytics`)

**Model Performance Dashboard:**

```
┌─────────────────────────────────────────────────────────┐
│ Model Performance Analytics                              │
├─────────────────────────────────────────────────────────┤
│ Filter: [All Genres ▼] [All Roles ▼] [Last 30 days ▼]  │
│                                                          │
│ ┌─────────────────────┐  ┌─────────────────────┐        │
│ │ Quality vs Speed    │  │ Top Models by Role  │        │
│ │ [scatter plot]      │  │ Writer: Celeste 8.2 │        │
│ │                     │  │ Arch: Qwen3-30B 8.5 │        │
│ └─────────────────────┘  └─────────────────────┘        │
│                                                          │
│ Recent Recommendations:                                  │
│ ├─ ✓ Applied: Switch continuity to DeepSeek-R1         │
│ ├─ ✗ Rejected: Lower writer temp to 0.7                │
│ └─ ? Pending: Try Dark Champion for romance            │
│                                                          │
│ [Export CSV]                                            │
└─────────────────────────────────────────────────────────┘
```

### Write Page Additions

**Mode Indicator:**
- Badge showing current mode name
- Quick-switch dropdown (doesn't require going to settings)
- Notification toast when auto-tuning applies changes

**Rating UI:**
- Optional 1-5 star rating after each chapter
- "Rate this chapter" prompt (dismissable)
- Cumulative rating visible in chapter list

---

## Workflow Integration

### Post-Chapter Scoring Hook

```python
# In StoryOrchestrator

async def _post_chapter_hook(
    self,
    chapter: Chapter,
    agent_outputs: dict[str, AgentOutput]
):
    """Called after each chapter completes the write-edit-continuity loop."""

    # 1. Collect performance metrics
    metrics = self._collect_metrics(agent_outputs)

    # 2. Run LLM quality judge (async, non-blocking)
    quality_task = asyncio.create_task(
        self.mode_service.judge_quality(chapter, self.story_state.brief)
    )

    # 3. Derive consistency score from continuity issues
    consistency_score = self._calculate_consistency_score(
        agent_outputs.get("continuity")
    )

    # 4. Wait for quality scores
    quality_scores = await quality_task

    # 5. Record everything to database
    await self.mode_service.record_generation(
        project_id=self.story_state.id,
        chapter_id=chapter.id,
        mode=self.current_mode,
        metrics=metrics,
        quality_scores=quality_scores,
        consistency_score=consistency_score
    )

    # 6. Check if tuning should run
    if self.mode_service.should_tune():
        recommendations = await self.mode_service.get_recommendations()
        await self._handle_recommendations(recommendations)

async def _handle_recommendations(
    self,
    recommendations: list[TuningRecommendation]
):
    """Process tuning recommendations based on autonomy level."""

    autonomy = self.settings.learning_autonomy

    for rec in recommendations:
        if autonomy == "manual":
            # Queue for user approval
            self.pending_recommendations.append(rec)
            self._emit("recommendation_pending", rec)

        elif autonomy == "cautious":
            if rec.recommendation_type == "temp_adjust":
                self._apply_recommendation(rec)
            else:
                self.pending_recommendations.append(rec)

        elif autonomy == "balanced":
            if rec.confidence > 0.8:
                self._apply_recommendation(rec)
            else:
                self.pending_recommendations.append(rec)

        elif autonomy in ("aggressive", "experimental"):
            self._apply_recommendation(rec)
            self._emit("recommendation_applied", rec)
```

---

## File Structure

```
services/
├── model_mode_service.py     # NEW: Mode management + scoring + learning
├── model_service.py          # Existing: Ollama operations
├── scoring_service.py        # NEW: Score collection and aggregation
└── __init__.py               # Add new services to container

memory/
├── mode_database.py          # NEW: SQLite operations for model_scores.db
└── ...

ui/pages/
├── settings.py               # Add mode configuration section
├── analytics.py              # NEW: Model performance dashboard
└── write.py                  # Add mode indicator, rating UI

docs/
├── MODELS.md                 # Update with new model research
└── plans/
    └── 2026-01-15-model-modes-scoring-design.md  # This document
```

---

## Implementation Phases

### Phase 1: Foundation
1. Create `model_scores.db` schema
2. Implement `ModelModeService` with preset modes
3. Add mode selector to Settings page
4. Wire up VRAM strategy (sequential/parallel/adaptive)

### Phase 2: Scoring
1. Implement `ScoreCollector` for implicit signals
2. Add LLM judge integration
3. Create post-chapter scoring hook in orchestrator
4. Add optional rating UI to Write page

### Phase 3: Learning
1. Implement tuning LLM integration
2. Create recommendation system
3. Add autonomy level settings
4. Implement auto-apply logic

### Phase 4: Analytics
1. Create Analytics page with charts
2. Add model performance aggregation
3. Implement CSV export
4. Add recommendation history view

### Phase 5: Polish
1. Update `docs/MODELS.md` with new research
2. Add mode quick-switch to Write page
3. Notification system for tuning events
4. Testing and refinement

---

## Verification

1. **Mode Switching:**
   - Switch between presets, verify correct models load
   - Create custom mode, verify persistence
   - Test VRAM strategies with different model sizes

2. **Scoring:**
   - Generate chapters, verify scores recorded in DB
   - Check LLM judge produces reasonable scores
   - Test implicit signals (regenerate, edit, accept)

3. **Learning:**
   - Generate 5+ chapters, verify recommendations appear
   - Test each autonomy level behavior
   - Verify recommendations respect VRAM/installed constraints

4. **Analytics:**
   - Verify charts render with real data
   - Test CSV export produces valid data
   - Check filtering works correctly

---

## Open Questions

1. **A/B Testing:** Should experimental mode generate same chapter with 2 models for direct comparison?
2. **Genre Detection:** Auto-detect genre from brief, or require user to specify?
3. **Cross-Project Learning:** Pool scores across all projects, or per-project isolation option?
4. **Model Warm-up:** Pre-load models in background before they're needed?

---

## Sources

- Claude Web Deep Research (January 2026)
- EQ-Bench Creative Writing Benchmark
- FlawedFictions Plot Hole Detection Research
- Agents' Room Framework (ICLR 2025)
