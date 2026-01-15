# Supported Models

This document lists all compatible models for Story Factory, with recommendations based on your hardware.

## Quick Start

```bash
# Default model - works on most gaming GPUs (8GB+ VRAM)
ollama pull huihui_ai/qwen3-abliterated:8b
```

## Model Recommendations by VRAM

### 8GB VRAM (RTX 3070, RTX 4060, etc.)

| Model | Size | Quantization | Notes |
|-------|------|--------------|-------|
| `huihui_ai/qwen3-abliterated:8b` | 8B | Q8 | **Recommended** - Great for creative writing |
| `dolphin-mistral:7b` | 7B | Q4 | Good general uncensored |
| `neural-chat:7b` | 7B | Q4 | Conversational style |

```bash
ollama pull huihui_ai/qwen3-abliterated:8b
ollama pull dolphin-mistral:7b
```

### 12GB VRAM (RTX 3080, RTX 4070, etc.)

| Model | Size | Quantization | Notes |
|-------|------|--------------|-------|
| `huihui_ai/qwen3-abliterated:14b` | 14B | Q8 | **Recommended** - Better quality |
| `dolphin-mixtral:8x7b` | 47B | Q4 | Excellent quality |
| `nous-hermes2:34b` | 34B | Q4 | Strong reasoning |

```bash
ollama pull huihui_ai/qwen3-abliterated:14b
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
| `huihui_ai/qwen3-abliterated:32b` | 32B | Q8 | ~20GB | **Recommended** - Excellent quality |
| `qwen2:72b` | 72B | Q4 | ~22GB | Excellent prose |

```bash
# Best quality for creative writing
ollama pull huihui_ai/qwen3-abliterated:32b
```

### 48GB+ VRAM (A6000, dual GPU, etc.)

| Model | Size | Notes |
|-------|------|-------|
| `llama3:70b` | 70B | Full precision possible |
| `command-r-plus:104b` | 104B | Exceptional quality |

## Uncensored Models

These models are uncensored and have fewer content restrictions:

| Model | Pull Command | VRAM Required |
|-------|--------------|---------------|
| Qwen3 8B Abliterated | `ollama pull huihui_ai/qwen3-abliterated:8b` | 8GB |
| Qwen3 14B Abliterated | `ollama pull huihui_ai/qwen3-abliterated:14b` | 12GB |
| Qwen3 32B Abliterated | `ollama pull huihui_ai/qwen3-abliterated:32b` | 24GB |
| Dolphin Mistral 7B | `ollama pull dolphin-mistral:7b` | 8GB |
| Dolphin Mixtral 8x7B | `ollama pull dolphin-mixtral:8x7b` | 12GB |

## Changing the Default Model

Edit `settings.json`:

```json
{
  "default_model": "huihui_ai/qwen3-abliterated:14b"
}
```

Or for different models per agent:

```json
{
  "use_per_agent_models": true,
  "agent_models": {
    "writer": "huihui_ai/qwen3-abliterated:32b",
    "editor": "huihui_ai/qwen3-abliterated:14b",
    "continuity": "huihui_ai/qwen3-abliterated:8b"
  }
}
```

## Performance Tips

1. **Close other GPU applications** before generating to free VRAM
2. **Use Q4 quantization** for best balance of quality and speed
3. **Start with smaller models** to test, then upgrade
4. **Monitor GPU usage** with `nvidia-smi` during generation

## Model Quality Comparison

Based on creative writing benchmarks:

```
Quality (creative writing):
Qwen3 32B          ████████████████████  Excellent
Qwen3 14B          ███████████████████   Very Good
Dolphin Mixtral    ████████████████      Very Good
Qwen3 8B           ██████████████        Good
Dolphin Mistral 7B ████████████          Good

Speed (tokens/sec on RTX 4090):
Qwen3 8B           ████████████████████  ~80 t/s
Dolphin Mistral 7B ████████████████████  ~85 t/s
Qwen3 14B          ██████████████████    ~50 t/s
Dolphin Mixtral    ██████████████        ~40 t/s
Qwen3 32B          ████████████          ~25 t/s
```

## Troubleshooting

### Model won't load
- Check VRAM: `nvidia-smi`
- Try a smaller quantization: Q3 instead of Q4
- Close other applications using GPU

### Slow generation
- Use a smaller model
- Reduce `max_tokens` in settings
- Ensure no other GPU processes running

### Poor quality output
- Try a larger model
- Increase temperature (0.8-1.0 for creative writing)
- Try different models - some are better for specific genres
