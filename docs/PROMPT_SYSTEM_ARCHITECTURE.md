# LLM Prompt System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User/Agent                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              EnhancedBaseAgent (Optional)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  generate_from_template(task, **vars)                │  │
│  │    1. Load template                                   │  │
│  │    2. Check cache                                     │  │
│  │    3. Generate (if cache miss)                        │  │
│  │    4. Record metrics                                  │  │
│  │    5. Cache response                                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────┬───────────────┬─────────────────┬────────────────────┘
      │               │                 │
      ▼               ▼                 ▼
┌──────────┐   ┌──────────┐    ┌──────────────┐
│Template  │   │Response  │    │  Metrics     │
│Manager   │   │Cache     │    │  Collector   │
└──────────┘   └──────────┘    └──────────────┘
      │               │                 │
      │               │                 │
      ▼               ▼                 ▼
┌──────────┐   ┌──────────┐    ┌──────────────┐
│Templates │   │Cache Dir │    │Metrics Dir   │
│(YAML)    │   │(.json)   │    │(.jsonl)      │
└──────────┘   └──────────┘    └──────────────┘
```

## Data Flow

### 1. Template-Based Generation

```
Request
  │
  ├─► PromptTemplateManager.load(agent, task, version)
  │     │
  │     ├─► Check cache (in-memory)
  │     │
  │     ├─► Load YAML file
  │     │     └─► Parse & validate
  │     │
  │     └─► Return PromptTemplate
  │
  ├─► template.render(**variables)
  │     │
  │     ├─► Validate variables
  │     └─► Return (system_prompt, user_prompt)
  │
  └─► Continue to generation...
```

### 2. Cached Generation

```
generate(prompt, model, temperature)
  │
  ├─► ResponseCache.get(prompt, model, temp)
  │     │
  │     ├─► Hash key = sha256(prompt|model|temp)[:16]
  │     │
  │     ├─► Check memory cache
  │     │     ├─► Hit? Check TTL
  │     │     │     ├─► Valid? Increment hit_count, return
  │     │     │     └─► Expired? Delete, continue
  │     │     └─► Miss? Continue
  │     │
  │     └─► Return None
  │
  ├─► LLM.generate(prompt)  [Cache miss - generate]
  │     └─► Get response
  │
  ├─► ResponseCache.put(prompt, model, temp, response)
  │     │
  │     ├─► Cache full? Evict LRU
  │     ├─► Store in memory
  │     └─► Save to disk (.json)
  │
  └─► Return response
```

### 3. Metrics Collection

```
After each generation:
  │
  ├─► Create PromptMetrics
  │     ├─► prompt_hash
  │     ├─► agent, task, model
  │     ├─► latency_ms
  │     ├─► validation_passed
  │     ├─► retry_count
  │     ├─► error (if any)
  │     └─► response_length
  │
  ├─► MetricsCollector.record(metrics)
  │     │
  │     ├─► Get daily log file (metrics_YYYY-MM-DD.jsonl)
  │     └─► Append JSON line
  │
  └─► Complete

Analysis (on-demand):
  │
  ├─► MetricsCollector.get_agent_stats(agent, days=7)
  │     │
  │     ├─► Load recent JSONL files
  │     ├─► Filter by agent
  │     └─► Aggregate:
  │           ├─► avg_latency_ms
  │           ├─► p95_latency_ms
  │           ├─► success_rate
  │           ├─► validation_pass_rate
  │           ├─► avg_retries
  │           ├─► error_rate
  │           ├─► cache_hit_rate
  │           └─► total_tokens_used
  │
  └─► Return statistics
```

## File Structure

```
story-factory/
├── prompts/
│   ├── __init__.py                    # Module exports
│   ├── template_manager.py            # Template system
│   ├── cache.py                       # Response caching
│   ├── metrics.py                     # Metrics collection
│   ├── integration_example.py         # Usage demo
│   ├── README.md                      # Documentation
│   └── templates/                     # YAML templates
│       ├── writer/
│       │   ├── write_chapter_v1.yaml
│       │   └── write_scene_v1.yaml
│       ├── editor/
│       │   └── edit_chapter_v1.yaml
│       └── architect/
│           └── create_characters_v1.yaml
│
├── output/
│   ├── cache/                         # Cached responses
│   │   ├── abc123.json                # {hash}.json
│   │   └── def456.json
│   └── metrics/                       # Metrics logs
│       ├── metrics_2025-01-20.jsonl   # Daily logs
│       └── metrics_2025-01-19.jsonl
│
├── docs/
│   ├── PROMPT_ARCHITECTURE.md         # Analysis document
│   └── PROMPT_IMPLEMENTATION_SUMMARY.md
│
└── tests/
    └── unit/
        ├── test_prompt_template_manager.py
        ├── test_prompt_cache.py
        └── test_prompt_metrics.py
```

## Component Interaction

```
┌─────────────────────────────────────────────────────────┐
│                    BaseAgent                             │
│  ┌────────────────────────────────────────────────┐    │
│  │  generate(prompt, context, temperature)         │    │
│  │    - Build messages                             │    │
│  │    - Rate limiting (semaphore)                  │    │
│  │    - Retry logic (max 3)                        │    │
│  │    - Validation (min length)                    │    │
│  │    - Error handling                             │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ can be enhanced with
                        ▼
┌─────────────────────────────────────────────────────────┐
│               EnhancedBaseAgent                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  generate_from_template(task, **vars)           │    │
│  │    - Load template ──────────┐                  │    │
│  │    - Render with vars        │                  │    │
│  │    - Check cache ────────────┼────┐             │    │
│  │    - Generate (via super)    │    │             │    │
│  │    - Record metrics ─────────┼────┼────┐        │    │
│  │    - Cache response          │    │    │        │    │
│  └─────────────────────────┬────┴────┴────┴────────┘    │
└────────────────────────────┼─────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │Template  │      │Cache     │      │Metrics   │
    │Manager   │      │          │      │Collector │
    └──────────┘      └──────────┘      └──────────┘
```

## Key Design Decisions

### 1. Template Format: YAML
**Why:** Human-readable, supports multi-line strings, widely used, good tooling

### 2. Cache Key: Hash(prompt + model + temperature)
**Why:** 
- Deterministic (same inputs = same key)
- Short (16 chars vs full prompt)
- Collision-resistant (SHA-256)
- Model/temp aware (different configs = different cache entries)

### 3. Cache Storage: Memory + Disk
**Why:**
- Memory for fast access during session
- Disk for persistence across restarts
- LRU eviction to limit memory usage
- TTL to prevent stale responses

### 4. Metrics Format: JSONL (JSON Lines)
**Why:**
- Append-only (efficient writes)
- One record per line (easy parsing)
- Human-readable (debugging)
- Compressible (gzip-friendly)
- Daily files (easy cleanup)

### 5. Shared Resources: Singleton Pattern
**Why:**
- One cache/metrics instance across all agents
- Reduced memory overhead
- Shared hit counts and statistics
- Consistent configuration

## Performance Characteristics

### Template Loading
- **First load:** ~5-10ms (YAML parse + validation)
- **Cached load:** ~0.1ms (in-memory lookup)
- **Memory per template:** ~2-5 KB

### Response Cache
- **Cache hit:** ~0.1ms (hash lookup)
- **Cache miss:** LLM latency (5-30s)
- **Memory per entry:** ~5-20 KB (depends on response size)
- **Disk I/O:** Async, non-blocking

### Metrics Collection
- **Record time:** ~1-2ms (append to file)
- **Disk per metric:** ~500 bytes (compressed)
- **Query time:** ~100-500ms (load + aggregate)

## Integration Points

### Current System
```python
# Before: Direct generation
response = agent.generate(prompt, context)
```

### With Templates Only
```python
# Load and render template
manager = PromptTemplateManager(template_dir)
template = manager.load("writer", "write_chapter")
system, user = template.render(chapter=1, title="...")
response = agent.generate(user, context)
```

### With Full System
```python
# Enhanced agent with all features
agent = EnhancedBaseAgent(...)
response = agent.generate_from_template(
    task="write_chapter",
    chapter=1,
    title="...",
    # ... template variables
)
# Automatic: template loading, caching, metrics
```

### Backward Compatible
```python
# Existing code continues to work
agent = WriterAgent()
response = agent.write_chapter(story_state, chapter)
# No changes needed!
```

## Monitoring & Observability

### Cache Statistics
```python
stats = cache.get_stats()
# {
#   'size': 150,
#   'max_size': 1000,
#   'utilization': 0.15,
#   'total_hits': 45,
#   'avg_age_seconds': 1200,
#   'oldest_age_seconds': 3400,
# }
```

### Agent Performance
```python
stats = collector.get_agent_stats("writer", days=7)
# {
#   'total_calls': 234,
#   'success_rate': 0.96,
#   'avg_latency_ms': 8500,
#   'p95_latency_ms': 15000,
#   'validation_pass_rate': 0.98,
#   'avg_retries': 0.2,
#   'error_rate': 0.04,
#   'cache_hit_rate': 0.32,
# }
```

### Model Comparison
```python
comparison = collector.get_model_comparison(days=7)
# [
#   {'model': 'dolphin3:8b', 'total_calls': 150, 'avg_latency_ms': 5000, 'success_rate': 0.98},
#   {'model': 'qwen3:30b', 'total_calls': 84, 'avg_latency_ms': 12000, 'success_rate': 0.96},
# ]
```

This architecture provides a solid foundation for prompt management while maintaining simplicity and backward compatibility.
