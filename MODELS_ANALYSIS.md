# Model Analysis: NSFW Creative Writing (January 2026)

## Current Setup Assessment

### Currently Installed: `tohur/natsumura-storytelling-rp-llama-3.1`

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Age** | Older | Based on Llama 3.1 (July 2024), not the newest |
| **Size** | 8B | Good for testing, not optimal for quality |
| **NSFW Tuning** | Good | Fine-tuned on RP/storytelling datasets |
| **Prose Quality** | Medium | Adequate for drafts, not publication-ready |
| **Context** | 128K native | Good, but Ollama defaults limit this |

**Verdict**: Good starter model, but **not the best available for your hardware**.

---

## Better Models for RTX 4090 (24GB VRAM)

### Tier 1: Best Quality (70B Class)

| Model | Pull Command | VRAM | Prose Quality | NSFW |
|-------|--------------|------|---------------|------|
| **Midnight Miqu 70B v1.5** | `ollama pull vanilj/midnight-miqu-70b-v1.5:Q4_K_M` | ~22GB | Excellent | Yes |
| **Llama 3.3 70B Abliterated** | `ollama pull huihui_ai/llama3.3-abliterated` | ~22GB | Excellent | Yes |
| **Lumimaid 70B** | `ollama pull lumimaid-v0.2:70b-q4_K_M` | ~22GB | Excellent | Yes |

#### Midnight Miqu 70B v1.5 (Recommended)
- **Base**: SLERP merge of Miqu-1-70B + Midnight-Rose-70B
- **Strengths**: Considered the **benchmark for pure prose quality**
- **Context**: 32K-64K tokens
- **NSFW**: Specifically designed for roleplaying and storytelling
- **Community**: Widely regarded as best for creative NSFW writing

#### Llama 3.3 70B Abliterated (Newest)
- **Base**: Meta's Llama 3.3 (December 2024) with abliteration
- **Strengths**: State-of-art 70B, similar to 405B performance
- **Context**: 128K tokens
- **NSFW**: Abliteration removes refusals
- **Note**: Newest architecture, may have better reasoning

### Tier 2: Excellent Quality (30-50B Class)

| Model | Pull Command | VRAM | Notes |
|-------|--------------|------|-------|
| **Qwen3 32B Abliterated** | `ollama pull huihui_ai/qwen3-abliterated:32b` | ~18GB | Very new, excellent reasoning |
| **Dolphin Mixtral 8x7B** | `ollama pull dolphin-mixtral:8x7b` | ~16GB | Eric Hartford's, proven quality |

### Tier 3: Good Quality (8-14B Class)

| Model | Pull Command | VRAM | Notes |
|-------|--------------|------|-------|
| **Qwen3 14B Abliterated** | `ollama pull huihui_ai/qwen3-abliterated:14b` | ~10GB | Newest uncensored, fast |
| **Lumimaid 8B** | `ollama pull lumimaid-v0.2:8b` | ~6GB | 60% RP/erotic training data |
| **Natsumura 8B** (current) | Already installed | ~5GB | Good, but older |

---

## Model Comparison: NSFW Writing Quality

Based on community benchmarks and reports:

```
Writing Quality for Erotic/NSFW Fiction:

Midnight Miqu 70B    ████████████████████ 10/10  "Benchmark for prose"
Llama 3.3 70B Abl.   ███████████████████  9.5/10 "Newest, excellent"
Lumimaid 70B         ██████████████████   9/10   "Purpose-built for RP"
Qwen3 32B Abl.       █████████████████    8.5/10 "Great reasoning"
Dolphin Mixtral      ████████████████     8/10   "Proven reliable"
Qwen3 14B Abl.       ██████████████       7/10   "Fast, good quality"
Natsumura 8B         ████████████         6/10   "Adequate starter"
```

---

## Critical: Ollama Configuration Issues

### Problem: Default Context is Too Low

Ollama defaults to **2048-4096 tokens** context, which is **way too small** for story writing. This means:
- Your story context gets **silently truncated**
- The model "forgets" earlier parts of the story
- Continuity suffers without you knowing

### Solution: Configure Context Properly

For your 24GB VRAM, you can use **32K-65K context**.

**Option 1: Per-request (current code does this)**
```python
# Already in our code via options parameter
options={"num_ctx": 32768}
```

**Option 2: Create custom model with higher context**
```bash
# Create a Modelfile
echo 'FROM vanilj/midnight-miqu-70b-v1.5:Q4_K_M
PARAMETER num_ctx 32768' > Modelfile

# Create the model
ollama create midnight-miqu-32k -f Modelfile
```

**Option 3: Environment variable**
```bash
set OLLAMA_NUM_CTX=32768
```

---

## Recommendations

### For Best Results on RTX 4090:

1. **Pull Midnight Miqu 70B** (best prose quality):
   ```bash
   ollama pull vanilj/midnight-miqu-70b-v1.5:Q4_K_M
   ```

2. **Or pull Llama 3.3 70B Abliterated** (newest):
   ```bash
   ollama pull huihui_ai/llama3.3-abliterated
   ```

3. **Update config.py**:
   ```python
   DEFAULT_MODEL = "vanilj/midnight-miqu-70b-v1.5:Q4_K_M"
   ```

4. **Add context configuration** to the code

### For Testing/Development (faster):
Keep Natsumura 8B for quick iterations, use 70B for final output.

---

## What "Abliterated" vs "Uncensored" Means

| Term | Meaning |
|------|---------|
| **Abliterated** | Refusal behavior surgically removed via activation steering |
| **Uncensored** | Trained without RLHF alignment (broader term) |
| **Fine-tuned for RP** | Specifically trained on roleplay/erotic datasets |

**Midnight Miqu** = Merge designed for creative writing + NSFW
**Abliterated models** = Base model with refusals removed
**Lumimaid** = 60% trained on RP/erotic content

---

## Sources

- [Ollama Uncensored Models](https://ollama.com/search?q=uncensored)
- [Midnight Miqu on HuggingFace](https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5)
- [Ollama Context Length Docs](https://docs.ollama.com/context-length)
- [Best Uncensored LLMs Guide](https://www.arsturn.com/blog/finding-the-best-uncensored-llm-on-ollama-a-deep-dive-guide)
- [The Sovereign Stack: Best Uncensored LLMs](https://www.watsonout.com/editorials/the-sovereign-stack-best-uncensored-llms-for-local-inference-dec-2025/)
- [Qwen3 Abliterated on Ollama](https://ollama.com/huihui_ai/qwen3-abliterated)
- [Llama 3.3 Abliterated on Ollama](https://ollama.com/huihui_ai/llama3.3-abliterated)
