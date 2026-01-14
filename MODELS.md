# Supported Models

This document lists all compatible models for Story Factory, with recommendations based on your hardware.

## Quick Start

```bash
# Default model - works on most gaming GPUs (8GB+ VRAM)
ollama pull tohur/natsumura-storytelling-rp-llama-3.1
```

## Model Recommendations by VRAM

### 8GB VRAM (RTX 3070, RTX 4060, etc.)

| Model | Size | Quantization | Notes |
|-------|------|--------------|-------|
| `tohur/natsumura-storytelling-rp-llama-3.1` | 8B | Q8 | **Recommended** - Tuned for storytelling |
| `dolphin-mistral:7b` | 7B | Q4 | Good general uncensored |
| `neural-chat:7b` | 7B | Q4 | Conversational style |

```bash
ollama pull tohur/natsumura-storytelling-rp-llama-3.1
ollama pull dolphin-mistral:7b
```

### 12GB VRAM (RTX 3080, RTX 4070, etc.)

| Model | Size | Quantization | Notes |
|-------|------|--------------|-------|
| `dolphin-mixtral:8x7b` | 47B | Q4 | Excellent quality |
| `nous-hermes2:34b` | 34B | Q4 | Strong reasoning |
| `yi:34b` | 34B | Q4 | Good creative writing |

```bash
ollama pull dolphin-mixtral:8x7b
```

### 16GB VRAM (RTX 4080, etc.)

| Model | Size | Quantization | Notes |
|-------|------|--------------|-------|
| `command-r:35b` | 35B | Q4 | Excellent instruction following |
| `mixtral:8x7b` | 47B | Q4 | High quality |

### 24GB VRAM (RTX 3090, RTX 4090)

| Model | Size | Quantization | VRAM Usage | Notes |
|-------|------|--------------|------------|-------|
| `vanilj/midnight-miqu-70b-v1.5:Q4_K_M` | 70B | Q4 | ~22GB | **Best for NSFW** |
| `vanilj/midnight-miqu-70b-v1.5:Q3_K_M` | 70B | Q3 | ~18GB | Lower quality, fits easier |
| `lumimaid-v0.2:70b-q4_K_M` | 70B | Q4 | ~22GB | Roleplay focused |
| `qwen2:72b` | 72B | Q4 | ~22GB | Excellent prose |

```bash
# Best quality for creative writing
ollama pull vanilj/midnight-miqu-70b-v1.5:Q4_K_M
```

### 48GB+ VRAM (A6000, dual GPU, etc.)

| Model | Size | Notes |
|-------|------|-------|
| `llama3:70b` | 70B | Full precision possible |
| `command-r-plus:104b` | 104B | Exceptional quality |

## NSFW-Capable Models

These models are uncensored and can generate adult content:

| Model | Pull Command | VRAM Required |
|-------|--------------|---------------|
| Natsumura 8B | `ollama pull tohur/natsumura-storytelling-rp-llama-3.1` | 8GB |
| Dolphin Mistral 7B | `ollama pull dolphin-mistral:7b` | 8GB |
| Dolphin Mixtral 8x7B | `ollama pull dolphin-mixtral:8x7b` | 12GB |
| Midnight Miqu 70B | `ollama pull vanilj/midnight-miqu-70b-v1.5:Q4_K_M` | 24GB |
| Lumimaid 70B | `ollama pull lumimaid-v0.2:70b-q4_K_M` | 24GB |

## Changing the Default Model

Edit `config.py`:

```python
# Default: 8B model for broad compatibility
DEFAULT_MODEL = "tohur/natsumura-storytelling-rp-llama-3.1"

# For RTX 4090 or similar:
DEFAULT_MODEL = "vanilj/midnight-miqu-70b-v1.5:Q4_K_M"
```

## Using Different Models for Different Agents

You can configure different models per agent in the orchestrator:

```python
from workflows.orchestrator import StoryOrchestrator

# Use a larger model for writing, smaller for editing
orchestrator = StoryOrchestrator()
orchestrator.writer.model = "vanilj/midnight-miqu-70b-v1.5:Q4_K_M"
orchestrator.editor.model = "tohur/natsumura-storytelling-rp-llama-3.1"
orchestrator.continuity.model = "tohur/natsumura-storytelling-rp-llama-3.1"
```

## Performance Tips

1. **Close other GPU applications** before generating to free VRAM
2. **Use Q4 quantization** for best balance of quality and speed
3. **Start with smaller models** to test, then upgrade
4. **Monitor GPU usage** with `nvidia-smi` during generation

## Model Quality Comparison

Based on creative writing benchmarks:

```
Quality (NSFW content):
Midnight Miqu 70B  ████████████████████  Excellent
Lumimaid 70B       ███████████████████   Excellent
Dolphin Mixtral    ████████████████      Very Good
Natsumura 8B       ██████████████        Good
Dolphin Mistral 7B ████████████          Good

Speed (tokens/sec on RTX 4090):
Natsumura 8B       ████████████████████  ~80 t/s
Dolphin Mistral 7B ████████████████████  ~85 t/s
Dolphin Mixtral    ██████████████        ~40 t/s
Midnight Miqu 70B  ████████              ~20 t/s
```

## Troubleshooting

### Model won't load
- Check VRAM: `nvidia-smi`
- Try a smaller quantization: Q3 instead of Q4
- Close other applications using GPU

### Slow generation
- Use a smaller model
- Reduce `MAX_TOKENS` in config.py
- Ensure no other GPU processes running

### Poor quality output
- Try a larger model
- Increase temperature (0.8-1.0 for creative writing)
- Try different models - some are better for specific genres
