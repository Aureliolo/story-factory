# Data Models and Schemas

<!-- Generated: 2026-01-24 | Updated: 2026-02-20 | Files scanned: 222 | Token estimate: ~900 -->

## Core Story Models (`memory/story_state.py`)

### StoryState
Complete story context, serialized to JSON.

```python
class StoryState(BaseModel):
    id: str                              # UUID
    created_at: datetime
    updated_at: datetime
    project_name: str
    project_description: str
    last_saved: datetime | None
    world_db_path: str                   # SQLite file path

    interview_history: list[dict[str, str]]
    reviews: list[dict[str, Any]]

    brief: StoryBrief | None
    world_description: str
    world_rules: list[str]
    characters: list[Character]
    plot_summary: str
    plot_points: list[PlotPoint]
    chapters: list[Chapter]
    current_chapter: int

    outline_variations: list[OutlineVariation]
    selected_variation_id: str | None

    timeline: list[str]
    established_facts: list[str]
    status: str  # interview, outlining, writing, editing, complete

    # Project-specific overrides (None = use global)
    target_chapters: int | None
    target_characters_min/max: int | None
    # ... more target_* fields
```

### StoryBrief
Initial story configuration from interview.

```python
class StoryBrief(BaseModel):
    premise: str
    genre: str
    subgenres: list[str]
    tone: str
    themes: list[str]
    setting_time: str
    setting_place: str
    target_length: TargetLength  # short_story, novella, novel
    language: str = "English"
    content_rating: str  # none, mild, moderate, explicit
    content_preferences: list[str]
    content_avoid: list[str]
    additional_notes: str
```

### Character
```python
class Character(BaseModel):
    name: str
    role: str  # protagonist, antagonist, supporting
    description: str
    personality_traits: list[str]
    goals: list[str]
    relationships: dict[str, str]  # char_name → relationship
    arc_notes: str
    arc_progress: dict[int, str]  # chapter → arc state
```

### Chapter
```python
class Chapter(BaseModel):
    number: int
    title: str
    outline: str
    content: str
    word_count: int
    status: str  # pending, drafted, edited, reviewed, final
    revision_notes: list[str]
    scenes: list[Scene]
    versions: list[ChapterVersion]
    current_version_id: str | None
```

### Scene
```python
class Scene(BaseModel):
    id: str
    title: str
    outline: str
    goal: str  # One-sentence purpose
    content: str
    word_count: int
    pov_character: str
    location: str
    beats: list[str]
    goals: list[str]  # Specific checkpoints
    order: int
    status: str
```

### PlotPoint
```python
class PlotPoint(BaseModel):
    description: str
    chapter: int | None
    completed: bool
    foreshadowing_planted: bool
```

### OutlineVariation
Alternative story structures for user selection.

```python
class OutlineVariation(BaseModel):
    id: str
    created_at: datetime
    name: str
    world_description: str
    world_rules: list[str]
    characters: list[Character]
    plot_summary: str
    plot_points: list[PlotPoint]
    chapters: list[Chapter]
    user_rating: int  # 0-5
    user_notes: str
    is_favorite: bool
    ai_rationale: str
```

## World Database Models (`memory/entities.py`)

### Entity
```python
class Entity(BaseModel):
    id: str
    type: str  # character, location, item, faction, concept
    name: str
    description: str
    attributes: dict[str, Any]  # includes lifecycle/temporal data
    created_at: datetime
    updated_at: datetime
```

### Relationship
```python
class Relationship(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation_type: str  # knows, loves, hates, located_in, owns, member_of
    description: str
    strength: float  # 0.0-1.0
    bidirectional: bool
    attributes: dict[str, Any]
    created_at: datetime
```

### WorldEvent
```python
class WorldEvent(BaseModel):
    id: str
    description: str
    chapter_number: int | None
    timestamp_in_story: str  # "Day 3", "Year 1042"
    consequences: list[str]
    created_at: datetime
```

### EventParticipant
```python
class EventParticipant(BaseModel):
    event_id: str
    entity_id: str
    role: str  # actor, location, affected, witness
```

## World Calendar Models (`memory/world_calendar.py`)

### CalendarMonth
```python
class CalendarMonth(BaseModel):
    name: str           # e.g., "Frostfall"
    days: int = 30
    description: str = ""
```

### HistoricalEra
```python
class HistoricalEra(BaseModel):
    name: str           # e.g., "Age of Dragons"
    start_year: int
    end_year: int | None = None
    description: str = ""
```

### WorldCalendar
```python
class WorldCalendar(BaseModel):
    id: str
    current_era_name: str
    era_abbreviation: str           # e.g., "AD", "TE"
    era_start_year: int = 1
    months: list[CalendarMonth]
    days_per_week: int = 7
    day_names: list[str] = []
    current_story_year: int
    story_start_year: int | None
    story_end_year: int | None
    eras: list[HistoricalEra]
    date_format: str = "{day} {month}, Year {year} {era}"

    def format_date(self, year, month=None, day=None) -> str
    def validate_date(self, year, month=None, day=None) -> bool
```

## World Settings Models (`memory/world_settings.py`)

### WorldSettings
```python
class WorldSettings(BaseModel):
    id: str
    calendar: WorldCalendar | None
    timeline_start_year: int | None
    timeline_end_year: int | None
    validate_temporal_consistency: bool = True
    created_at: datetime
    updated_at: datetime
```

## World Health Models (`memory/world_health.py`)

### WorldHealthMetrics
```python
class WorldHealthMetrics(BaseModel):
    # Entity counts
    total_entities: int
    entity_counts: dict[str, int]
    total_relationships: int

    # Orphan detection
    orphan_count: int
    orphan_entities: list[dict]

    # Circular relationship detection
    circular_count: int
    circular_relationships: list[dict]

    # Quality metrics
    average_quality: float  # 0-10
    quality_distribution: dict[str, int]
    low_quality_entities: list[dict]

    # Contradiction detection
    contradiction_count: int
    contradictions: list[dict]

    # Temporal consistency
    temporal_error_count: int
    temporal_warning_count: int
    average_temporal_consistency: float  # 0-10
    temporal_issues: list[dict]

    # Computed metrics
    relationship_density: float
    health_score: float  # 0-100
    recommendations: list[str]

    def calculate_health_score(self) -> float
```

## Generation Mode Models (`memory/mode_models.py`)

### Enums

```python
class VramStrategy(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

class LearningTrigger(str, Enum):
    OFF = "off"
    AFTER_PROJECT = "after_project"
    PERIODIC = "periodic"
    CONTINUOUS = "continuous"

class AutonomyLevel(str, Enum):
    MANUAL = "manual"
    CAUTIOUS = "cautious"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"

class ModelSizeTier(str, Enum):
    LARGE = "large"    # 20GB+
    MEDIUM = "medium"  # 8-20GB
    SMALL = "small"    # 3-8GB
    TINY = "tiny"      # <3GB
```

### GenerationMode

```python
class GenerationMode(BaseModel):
    id: str
    name: str
    description: str
    agent_models: dict[str, str]
    agent_temperatures: dict[str, float]
    vram_strategy: VramStrategy
    is_preset: bool
    is_experimental: bool
```

### Scoring Models

```python
class QualityScores(BaseModel):
    prose_quality: float | None
    instruction_following: float | None
    consistency_score: float | None

class PerformanceMetrics(BaseModel):
    tokens_generated: int | None
    time_seconds: float | None
    tokens_per_second: float | None
    vram_used_gb: float | None

class GenerationScore(BaseModel):
    project_id: str
    chapter_id: str | None
    agent_role: str
    model_id: str
    mode_name: str
    quality: QualityScores
    performance: PerformanceMetrics
    signals: ImplicitSignals
    timestamp: datetime
```

### Preset Modes

| Mode | Description | VRAM Strategy |
|------|-------------|---------------|
| `quality_max` | Largest models, sequential | SEQUENTIAL |
| `quality_creative` | High temp writer | SEQUENTIAL |
| `balanced` | Medium models | ADAPTIVE |
| `draft_fast` | Smaller models | PARALLEL |
| `experimental` | Varies for data | ADAPTIVE |

## World Quality Models (`memory/world_quality/`) ~1,544 lines

### Entity Quality Scores (0-10 scale)

| Model | Dimensions |
|-------|------------|
| `CharacterQualityScores` | depth, goals, flaws, uniqueness, arc_potential |
| `LocationQualityScores` | atmosphere, significance, story_relevance, distinctiveness |
| `RelationshipQualityScores` | tension, dynamics, story_potential, authenticity |
| `FactionQualityScores` | coherence, influence, conflict_potential, distinctiveness |
| `ItemQualityScores` | significance, uniqueness, narrative_potential, integration |
| `ConceptQualityScores` | relevance, depth, manifestation, resonance |
| `CalendarQualityScores` | internal_consistency, thematic_fit, completeness, uniqueness |
| `EventQualityScores` | significance, temporal_plausibility, causal_coherence, narrative_potential, entity_integration |

All quality score classes provide:
- `average` property → float
- `to_dict()` → dict[str, float | str]
- `weak_dimensions(threshold=7.0)` → list[str]

### RefinementConfig

```python
class RefinementConfig(BaseModel):
    max_iterations: int = 3
    quality_threshold: float = 7.5
    creator_temperature: float = 0.9
    judge_temperature: float = 0.1
    refinement_temperature: float = 0.7
```

## Temporal Validation Models

### TemporalValidationIssue
```python
class TemporalValidationIssue(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str
    error_type: TemporalErrorType  # predates_dependency, invalid_era, anachronism,
                                   # post_destruction, invalid_date, lifespan_overlap, founding_order
    severity: TemporalErrorSeverity  # warning, error
    message: str
    related_entity_id: str | None
    related_entity_name: str | None
    suggestion: str
```

## Database Schemas

> Schemas shown are simplified. See source files for complete schemas.

### WorldDatabase SQLite Tables

**entities**
```sql
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    attributes TEXT DEFAULT '{}',  -- JSON (includes lifecycle/temporal data)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

**relationships**
```sql
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES entities(id),
    target_id TEXT NOT NULL REFERENCES entities(id),
    relation_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    strength REAL DEFAULT 1.0,
    bidirectional INTEGER DEFAULT 0,
    attributes TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
)
```

**events**
```sql
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    chapter_number INTEGER,
    timestamp_in_story TEXT DEFAULT '',
    consequences TEXT DEFAULT '[]',
    created_at TEXT NOT NULL
)
```

**event_participants**
```sql
CREATE TABLE event_participants (
    event_id TEXT NOT NULL REFERENCES events(id),
    entity_id TEXT NOT NULL REFERENCES entities(id),
    role TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_id)
)
```

**entity_versions**
```sql
CREATE TABLE entity_versions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(id),
    version_number INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    change_type TEXT NOT NULL CHECK(change_type IN ('created','refined','edited','regenerated')),
    change_reason TEXT DEFAULT '',
    quality_score REAL DEFAULT NULL
)
```

### ModeDatabase SQLite Tables

**generation_scores**
```sql
CREATE TABLE generation_scores (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    chapter_id TEXT,
    agent_role TEXT NOT NULL,
    model_id TEXT NOT NULL,
    mode_name TEXT NOT NULL,
    prose_quality REAL,
    instruction_following REAL,
    consistency_score REAL,
    tokens_generated INTEGER,
    time_seconds REAL,
    tokens_per_second REAL,
    vram_used_gb REAL,
    timestamp TEXT NOT NULL
)
```

**world_entity_scores**
```sql
CREATE TABLE world_entity_scores (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    model_id TEXT NOT NULL,
    quality_scores TEXT,             -- JSON
    iteration INTEGER,
    timestamp TEXT NOT NULL,
    temporal_consistency_score REAL,
    temporal_validation_errors TEXT  -- JSON array
)
```

**prompt_metrics** / **custom_modes** (see `mode_database/_schema.py`)

## Settings Schema (`settings/`)

### Settings Dataclass (~100+ fields)

**Connection:**
- `ollama_url`, `context_size`, `max_tokens`, `ollama_timeout`

**Model Selection:**
- `default_model`, `use_per_agent_models`, `agent_models`, `custom_model_tags`

**Temperatures:**
- `agent_temperatures`, `temp_brief_extraction`, `temp_edit_suggestions`, etc.

**World Generation:**
- `world_gen_*_min/max` for characters, locations, factions, items, concepts, relationships

**Quality:**
- `world_quality_enabled`, `world_quality_max_iterations`, `world_quality_threshold`

**Temporal:**
- `validate_temporal_consistency`, `generate_calendar_on_world_build`

**RAG:**
- `embedding_model` (auto-migrated if empty)

## Workflow Events (`services/orchestrator/`)

```python
@dataclass
class WorkflowEvent:
    event_type: str  # agent_start, agent_complete, user_input_needed, progress, error
    agent_name: str
    message: str
    data: dict[str, Any] | None
    timestamp: datetime | None
    correlation_id: str | None
    phase: str | None
    progress: float | None  # 0.0-1.0
    chapter_number: int | None
    eta_seconds: float | None
```

## File Storage

- **Stories:** `output/stories/{uuid}.json` (full StoryState)
- **Worlds:** `output/worlds/{uuid}.sqlite` (WorldDatabase)
- **Backups:** `output/backups/{uuid}_{timestamp}.zip` (story + world bundled)
- **Logs:** `output/logs/story_factory.log` (rotating file handler)
- **Settings:** `src/settings.json` (gitignored, auto-backup in `.settings_backup/`)
