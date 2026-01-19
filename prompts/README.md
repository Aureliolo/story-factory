# Prompt Management System

This directory contains the prompt template management system for Story Factory.

## Components

### Template Manager (`template_manager.py`)
- Load versioned prompt templates from YAML files
- Manage few-shot examples
- Cache templates for performance
- Support for template variables and validation rules

### Response Cache (`cache.py`)
- LRU cache for LLM responses with TTL
- Disk persistence across sessions
- Automatic eviction of expired entries
- Hit counting and statistics

### Metrics Collector (`metrics.py`)
- Track prompt execution metrics (latency, tokens, errors)
- Aggregate statistics by agent, task, and model
- Daily JSONL log files for historical analysis
- Performance comparison across models

## Template Structure

Templates are stored in `templates/{agent}/{task}_v{version}.yaml`:

```yaml
version: "1.0"
agent: "writer"
task: "write_chapter"
system_prompt: "You are a writer..."
user_prompt_template: "Write chapter {number}: {title}"
variables:
  number: "int - Chapter number"
  title: "str - Chapter title"
validation:
  min_length: 1000
  max_length: 5000
examples:
  - context: "Fantasy adventure"
    input:
      number: 1
      title: "The Beginning"
    output: "Once upon a time..."
```

## Usage

### Loading Templates

```python
from prompts import PromptTemplateManager

manager = PromptTemplateManager(Path("prompts/templates"))
template = manager.load("writer", "write_chapter", version="latest")

system, user = template.render(
    number=1,
    title="The Beginning",
    # ... other variables
)
```

### Using the Cache

```python
from prompts import ResponseCache

cache = ResponseCache(Path("cache"), max_size=1000, ttl_seconds=3600)

# Check cache before generating
cached = cache.get(prompt, model, temperature)
if cached:
    return cached

# Generate and cache
response = llm.generate(prompt)
cache.put(prompt, model, temperature, response)
```

### Collecting Metrics

```python
from prompts import MetricsCollector, PromptMetrics
from datetime import datetime

collector = MetricsCollector(Path("metrics"))

# Record metrics after each LLM call
metrics = PromptMetrics(
    prompt_hash=hash_prompt(prompt),
    agent="writer",
    task="write_chapter",
    model=model,
    temperature=temperature,
    timestamp=datetime.now(),
    latency_ms=duration * 1000,
    validation_passed=True,
    retry_count=0,
    response_length=len(response),
)
collector.record(metrics)

# Analyze performance
stats = collector.get_agent_stats("writer", days=7)
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
```

## Benefits

1. **Separation of Concerns**: Prompts are separate from code
2. **Versioning**: Track prompt changes over time
3. **Performance**: Cache identical requests to save tokens and time
4. **Observability**: Comprehensive metrics for debugging and optimization
5. **Maintainability**: Templates can be updated without code changes
6. **Testing**: Easy to test different prompt variations

## Future Enhancements

- Prompt A/B testing framework
- Automatic prompt optimization based on quality metrics
- Few-shot example selection based on similarity
- Template inheritance and composition
- Prompt quality scoring
- Integration with BaseAgent for seamless usage
