# Data Models and Schemas

> Generated: 2026-01-24 | Freshness: Current

## Core Story Models (`memory/story_state.py`)

### StoryState (`story_state.py:404-622`)
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

### StoryBrief (`story_state.py:344-359`)
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

### Character (`story_state.py:15-66`)
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

### Chapter (`story_state.py:132-341`)
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

### Scene (`story_state.py:90-118`)
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

### PlotPoint (`story_state.py:81-87`)
```python
class PlotPoint(BaseModel):
    description: str
    chapter: int | None
    completed: bool
    foreshadowing_planted: bool
```

### OutlineVariation (`story_state.py:362-401`)
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

### Entity (`entities.py:9-18`)
```python
class Entity(BaseModel):
    id: str
    type: str  # character, location, item, faction, concept
    name: str
    description: str
    attributes: dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

### Relationship (`entities.py:21-32`)
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

### WorldEvent (`entities.py:35-43`)
```python
class WorldEvent(BaseModel):
    id: str
    description: str
    chapter_number: int | None
    timestamp_in_story: str  # "Day 3", "Year 1042"
    consequences: list[str]
    created_at: datetime
```

### EventParticipant (`entities.py:46-51`)
```python
class EventParticipant(BaseModel):
    event_id: str
    entity_id: str
    role: str  # actor, location, affected, witness
```

## Database Schema (`memory/world_database.py`)

### SQLite Tables

**entities**
```sql
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    attributes TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

**relationships**
```sql
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    strength REAL DEFAULT 1.0,
    bidirectional INTEGER DEFAULT 0,
    attributes TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
)
```

**events**
```sql
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    chapter_number INTEGER,
    timestamp_in_story TEXT DEFAULT '',
    consequences TEXT DEFAULT '[]',  -- JSON array
    created_at TEXT NOT NULL
)
```

**event_participants**
```sql
CREATE TABLE event_participants (
    event_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_id),
    FOREIGN KEY (event_id) REFERENCES events(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
)
```

**schema_version**
```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY
)
```

## Settings Schema (`settings.py`)

### Settings Dataclass
~100+ fields with validation. Key groups:

**Connection:**
- `ollama_url`: str = "http://localhost:11434"
- `context_size`: int = 32768
- `max_tokens`: int = 8192
- `ollama_timeout`: int = 120

**Model Selection:**
- `default_model`: str = "auto"
- `use_per_agent_models`: bool = True
- `agent_models`: dict[str, str]  # role → model_id
- `custom_model_tags`: dict[str, list[str]]  # model → roles

**Temperatures:**
- `agent_temperatures`: dict[str, float]
- `temp_brief_extraction`, `temp_edit_suggestions`, etc.

**World Generation:**
- `world_gen_characters_min/max`
- `world_gen_locations_min/max`
- `world_gen_factions_min/max`
- `world_gen_items_min/max`
- `world_gen_concepts_min/max`
- `world_gen_relationships_min/max`

**Quality:**
- `world_quality_enabled`: bool
- `world_quality_max_iterations`: int = 3
- `world_quality_threshold`: float = 7.0

### ModelInfo TypedDict (`settings.py:28-39`)
```python
class ModelInfo(TypedDict):
    name: str
    size_gb: float
    vram_required: int
    quality: int | float
    speed: int
    uncensored: bool
    description: str
    tags: list[str]  # Role suitability
```

## Workflow Events (`workflows/orchestrator.py:33-49`)

```python
@dataclass
class WorkflowEvent:
    event_type: str  # agent_start, agent_complete, user_input_needed, progress, error
    agent_name: str
    message: str
    data: dict[str, Any] | None
    timestamp: datetime | None
    correlation_id: str | None
    phase: str | None  # interview, architect, writer, editor, continuity
    progress: float | None  # 0.0-1.0
    chapter_number: int | None
    eta_seconds: float | None
```

## UI State (`ui/state.py`)

### ActionType Enum (`state.py:15-34`)
```python
class ActionType(Enum):
    ADD_ENTITY = "add_entity"
    DELETE_ENTITY = "delete_entity"
    UPDATE_ENTITY = "update_entity"
    ADD_RELATIONSHIP = "add_relationship"
    DELETE_RELATIONSHIP = "delete_relationship"
    UPDATE_RELATIONSHIP = "update_relationship"
    UPDATE_CHAPTER_CONTENT = "update_chapter_content"
    DELETE_CHAPTER = "delete_chapter"
    ADD_CHAPTER = "add_chapter"
    UPDATE_CHAPTER_FEEDBACK = "update_chapter_feedback"
    UPDATE_SETTINGS = "update_settings"
```

### UndoAction (`state.py:37-42`)
```python
@dataclass
class UndoAction:
    action_type: ActionType
    data: dict[str, Any]
    inverse_data: dict[str, Any]
```

## File Storage

**Stories:** `output/stories/{uuid}.json`
- Full StoryState serialized

**Worlds:** `output/worlds/{uuid}.sqlite`
- WorldDatabase with entities/relationships

**Backups:** `output/backups/{uuid}_{timestamp}.zip`
- Story JSON + World SQLite bundled

**Logs:** `logs/story_factory.log`
- Rotating file handler
