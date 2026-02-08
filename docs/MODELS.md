# Model Guide for Story Factory

This document provides comprehensive guidance on selecting the best LLM models for Story Factory's multi-agent story generation system.

**Last Updated:** February 2026
**Hardware Reference:** RTX 4090 (24GB VRAM)

## Quick Navigation

- 📚 [Quick Start](#quick-start) - Get running in 5 minutes
- 🎭 [Agent Roles](#agent-roles--requirements) - What each agent needs
- 💎 [Recommended Models](#recommended-model-stack) - Best combinations
- 🔬 [Model Deep Dive](#model-deep-dive) - Detailed model information
- 📊 [Research Findings](#research-findings) - Scientific backing
- 🛠️ [Troubleshooting](#troubleshooting) - Common issues

## Agent Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Story Generation Pipeline                  │
│                                                               │
│  👤 Interviewer (Fast, Conversational)                       │
│   │ Temp: 0.7  │  Model: Dolphin 8B                          │
│   └──────┬──────────────────────────────────────────────────│
│          ▼                                                    │
│  🏗️  Architect (Logical, Structured)                        │
│   │ Temp: 0.85 │  Model: Qwen3-30B or Llama 70B             │
│   └──────┬──────────────────────────────────────────────────│
│          ▼                                                    │
│  ┌──────────────────── Revision Loop ─────────────────────┐ │
│  │                                                          │ │
│  │  ✍️  Writer (Creative, Vivid)                           │ │
│  │   │ Temp: 0.9  │  Model: Celeste 12B                    │ │
│  │   └──────┬──────────────────────────────────────────    │ │
│  │          ▼                                                │ │
│  │  📝 Editor (Precise, Polished)                          │ │
│  │   │ Temp: 0.6  │  Model: Same as Writer                 │ │
│  │   └──────┬──────────────────────────────────────────    │ │
│  │          ▼                                                │ │
│  │  🔍 Continuity (Analytical, Strict)                     │ │
│  │   │ Temp: 0.3  │  Model: DeepSeek-R1 or Qwen3-30B      │ │
│  │   └──────┬──────────────────────────────────────────    │ │
│  │          │                                                │ │
│  │   Issues Found?  ────Yes──► Back to Writer (max 3x)     │ │
│  │          │                                                │ │
│  └──────────┼No───────────────────────────────────────────┘ │
│             ▼                                                 │
│  ✅ Final Chapter → Next Chapter                            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Model Selection Matrix

```
           │ Speed │ Quality │ Creativity │ VRAM  │ Best For
───────────┼───────┼─────────┼────────────┼───────┼─────────────
Dolphin 8B │  ⚡⚡⚡  │   ⭐⭐⭐   │     ⭐⭐     │  8GB  │ Interviewer
Celeste12B │  ⚡⚡   │  ⭐⭐⭐⭐  │    ⭐⭐⭐⭐    │ 14GB  │ Writer
Qwen3-30B  │  ⚡⚡   │  ⭐⭐⭐⭐⭐ │     ⭐⭐     │ 18GB  │ Architect
DeepSeek-R1│   ⚡   │  ⭐⭐⭐⭐⭐ │      ⭐     │ 10GB  │ Continuity
Llama 3.3  │   ⚡   │  ⭐⭐⭐⭐⭐ │     ⭐⭐     │ 42GB+ │ Premium
```

## Table of Contents

- [Quick Start](#quick-start)
- [Agent Roles & Requirements](#agent-roles--requirements)
- [Recommended Model Stack](#recommended-model-stack)
- [Model Deep Dive](#model-deep-dive)
- [HuggingFace Models](#huggingface-models-manual-install)
- [Quantization Guide](#quantization-guide)
- [Research Findings](#research-findings)
- [Sources](#sources)

---

## Quick Start

```bash
# Essential models for 24GB VRAM setup (January 2026)
ollama pull huihui_ai/qwen3-abliterated:30b       # NEW: Best reasoning (MoE, 18GB)
ollama pull huihui_ai/dolphin3-abliterated:8b     # Fast default, interviewer
ollama pull vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0  # Creative writing
```

---

## Agent Roles & Requirements

Story Factory uses 6 specialized agents, each with different model requirements:

| Agent | Purpose | Key Requirements | Priority |
|-------|---------|------------------|----------|
| **Writer** | Generate prose, dialogue, scenes | Creative, vivid, emotionally engaging, NSFW capable | Quality > Speed |
| **Architect** | Plan story structure, chapters, arcs | Logical reasoning, structure, coherence | Reasoning > Creativity |
| **Editor** | Polish, refine, improve prose | Precision, consistency, instruction-following | Balance |
| **Continuity** | Check plot holes, consistency | Strong reasoning, contradiction detection | Reasoning critical |
| **Interviewer** | Gather story requirements from user | Conversational, compliant, fast | Speed > Quality |
| **Judge** | Score entity quality | Independent numeric assessment, JSON output | Accuracy > Speed |

---

## Recommended Model Stack

### For 24GB VRAM (RTX 4090) - January 2026 Update

| Role | Model | Source | VRAM | Why This Model |
|------|-------|--------|------|----------------|
| **Architect** | Qwen3-30B-A3B Abliterated | Ollama | 18GB | **NEW**: MoE (30B/3B active), matches 70B reasoning at half VRAM |
| **Writer** | Celeste V1.9 12B | Ollama | 14GB | Purpose-built for fiction, OOC steering, excellent NSFW |
| **Writer (Alt)** | Dark Champion 21B MOE V2 | HuggingFace | 14GB | "OFF THE SCALE" prose, adjustable experts |
| **Editor** | Same as Writer (temp 0.5) | - | - | Maintains voice consistency |
| **Continuity** | DeepSeek-R1-Distill-Qwen-14B | HuggingFace | 10GB | **NEW**: Explicit `<think>` reasoning chains |
| **Interviewer** | Dolphin 3.0 8B | Ollama | 5GB | Fast, compliant, highly steerable |

**Key Change from 2025:** Qwen3-30B-A3B replaces Llama 3.3 70B for reasoning tasks - same quality at half the VRAM, enabling parallel model loading.

### For 16GB VRAM

| Role | Model | VRAM |
|------|-------|------|
| **Writer** | Celeste V1.9 12B or Lyra-Gutenberg 12B | 10GB |
| **Architect** | Qwen3-14B (hybrid thinking) | 10GB |
| **Continuity** | DeepSeek-R1-Qwen3-8B | 5GB |
| **Interviewer** | Dolphin 3.0 8B | 5GB |

### For 8GB VRAM

| Role | Model | Notes |
|------|-------|-------|
| **All roles** | Dolphin 3.0 8B | Best all-rounder at this size |
| **Writer (Alt)** | Gemma The Writer 9B | Specialized for fiction |

---

## Model Deep Dive

### Creative Writing Models (Writer Role)

#### Celeste V1.9 (MN-12B-Celeste)
- **Base:** Mistral NeMo 12B
- **Training:** Reddit Writing Prompts, Kalo's Opus 25K Instruct
- **Strengths:** OOC steering ("OOC: character should be more assertive"), excellent NSFW, steerable at any point
- **Context:** 8K native, extendable to 16K+
- **Ollama:** `vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0`

#### Lyra-Gutenberg 12B (NEW)
- **Base:** Mistral NeMo 12B
- **Training:** Gutenberg literature + Sao10K's Lyra improvements
- **Strengths:** "Peak of Gutenberg lineage", literary prose style, top UGI ranking
- **Note:** Use Mistral chat format, NOT ChatML
- **HuggingFace:** `nbeerbower/Lyra-Gutenberg-mistral-nemo-12B`

#### Dark Champion 21B MOE V2 (NEW)
- **Architecture:** 8x4B Mixture of Experts (21B total, was 18B in V1)
- **Strengths:** "OFF THE SCALE" prose quality, 50+ tok/s at 2 experts
- **Feature:** Adjustable expert count (2-8) for quality/speed tradeoff
- **Cons:** MOE can be inconsistent, occasional verbosity
- **HuggingFace:** `DavidAU/Llama-3.2-8X4B-MOE-V2-Dark-Champion-Instruct-uncensored-abliterated-21B-GGUF`

#### Midnight-Miqu 70B v1.5 (Premium)
- **Base:** Miqu (Mistral-based)
- **Strengths:** "Writes like a novelist", 32K context, emotional depth + narrative coherence
- **VRAM:** 42GB+ at Q4_K_M (requires offloading on single 4090)
- **Ollama:** `vanilj/midnight-miqu-70b-v1.5`

#### Gemma The Writer Collection
- **Variants:** Mighty-Sword-9B (32-bit mastered), DEADLINE-10B (2x longer outputs)
- **Strengths:** Reduced GPT-isms, varied prose, excellent instruction following
- **HuggingFace:** `DavidAU/Gemma-The-Writer-Mighty-Sword-9B-GGUF`

### Reasoning Models (Architect/Continuity Roles)

#### Qwen3-30B-A3B Abliterated (NEW - Recommended)
- **Architecture:** MoE - 30B total, only 3B active per token
- **Strengths:** Frontier reasoning at consumer VRAM, hybrid thinking mode
- **Feature:** Hybrid thinking mode (`<think>` tags stripped automatically)
- **VRAM:** ~18GB at Q4_K_M
- **Context:** 128K tokens
- **Ollama:** `huihui_ai/qwen3-abliterated:30b`

#### DeepSeek-R1-Distill-Qwen-14B (NEW)
- **Strengths:** Explicit `<think>` reasoning chains, excellent for plot hole detection
- **Benchmark:** R1-0528 achieved 87.5% on AIME 2025
- **Best for:** Continuity checking where you need to see the reasoning
- **HuggingFace:** `mradermacher/DeepSeek-R1-Distill-Qwen-14B-GGUF`

#### DeepSeek-R1-0528-Qwen3-8B (NEW)
- **Strengths:** SOTA reasoning in 8B package
- **Best for:** Fast continuity passes, reasoning on limited VRAM
- **HuggingFace:** `unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF`

#### Llama 3.3 70B Abliterated (Legacy)
- **Status:** Still excellent, but Qwen3-30B-A3B matches quality at half VRAM
- **Use case:** When you need the absolute best reasoning and have VRAM to spare
- **Ollama:** `huihui_ai/llama3.3-abliterated:70b-instruct-q4_K_M`

### General Purpose Models

#### Dolphin 3.0 8B (Eric Hartford)
- **Strengths:** Highly steerable via system prompts, excellent instruction following, no refusals
- **Best for:** Interviewer role, fast tasks, coding/math
- **Personality:** "Less about personality, all about raw, unfiltered smarts"
- **Ollama:** `huihui_ai/dolphin3-abliterated:8b`

#### Dolphin Mistral Nemo 12B
- **Strengths:** 128K context window, good for editing and refinement
- **Ollama:** `CognitiveComputations/dolphin-mistral-nemo:12b`

### Judge Models (Quality Scoring Role)

The judge role evaluates entity quality during world generation, producing calibrated
numeric scores in JSON. This requires models that can independently assess content
rather than copy patterns from the prompt.

**Critical finding (Issue #228, Feb 2026):** Hardcoded example scores in judge prompts
(e.g. `"coherence": 6.7`) caused most models to copy those exact values verbatim
instead of evaluating independently. Switching to parametric placeholders
(`"coherence": <float 0-10>`) eliminated copying across all tested models and
dramatically improved accuracy.

#### Benchmark Results

Tested 16 models via `scripts/evaluate_judge_accuracy.py` — 10 hand-crafted samples
at 3 quality tiers (terrible ~2-3, mediocre ~5, excellent ~8-9) with ground-truth
scores, measuring MAE (Mean Absolute Error), Spearman rank correlation, example
copying rate, and score spread.

**Parametric variant (production format, no example scores):**

| Rank | Model | Params | MAE | Rank Corr | Spread | Verdict |
|------|-------|--------|-----|-----------|--------|---------|
| 1 | **Gemma 3 12B** | 12B | **1.58** | **0.98** | 3.5 | Best judge |
| 2 | **Phi-4 14B** | 14B | **2.00** | **0.99** | 2.5 | Near-perfect ranking |
| 3 | Phi-4 Mini 3.8B | 3.8B | 2.04 | 0.89 | 2.6 | Best small judge |
| 4 | Gemma 3 4B | 4B | 2.04 | 0.94 | 2.2 | Good for size |
| 5 | Qwen3 30B Ablit (MoE) | 30B | 2.08 | 0.98 | 1.8 | Excellent |
| 6 | Celeste 12B | 12B | 2.19 | 0.78 | 1.8 | Usable |
| 7 | Qwen3 8B Ablit | 8B | 2.36 | 0.90 | 1.8 | Marginal |
| 8 | Dolphin3 8B Ablit | 8B | 2.38 | 0.90 | 1.8 | Marginal |
| 9 | Dolphin Mistral Nemo 12B | 12B | 2.46 | 0.69 | 1.9 | Marginal |
| 10 | Gemma 3 1B | 1B | 2.55 | 0.09 | 1.4 | Not viable |
| 11 | Dark Champion MoE 18B | 18B | 2.65 | 0.61 | 1.2 | Not viable |
| 12 | Qwen3 4B | 4B | 2.67 | 0.50 | 0.0 | Not viable (timeouts) |
| 13 | SmolLM2 1.7B | 1.7B | 2.95 | 0.26 | 1.0 | Not viable |
| 14 | Qwen3 0.6B | 0.6B | 3.04 | 0.28 | 0.6 | Not viable |
| - | Llama 3.3 70B (both) | 70B | FAIL | - | - | Too slow for 24GB |

**Key takeaways:**
- **Removing example scores from prompts is critical** — copy rates dropped from 65-100% to 0-8% for every model
- **All models overrate bad content** — even Gemma 12B scores terrible entities ~2.4 points too high (LLM positivity bias)
- **Excellent content is scored accurately** — top models achieve MAE < 0.6 for high-quality entities
- **Minimum viable judge is ~4B** — Gemma 3 4B and Phi-4 Mini (3.8B) both achieve rank correlation > 0.89
- **Size alone doesn't determine judge quality** — Dark Champion 18B MoE and Dolphin Mistral Nemo 12B are worse judges than Gemma 3 4B
- **70B models are impractical** as judges on 24GB VRAM — can't respond within 30s timeout

#### Recommended Judge Setup

| VRAM | Model | Why |
|------|-------|-----|
| 24GB | Gemma 3 12B | Best accuracy, near-perfect ranking |
| 16GB | Qwen3 30B MoE or Phi-4 14B | MoE fast inference; Phi-4 perfect rank correlation (0.99) |
| 8GB | Dolphin 3 8B or Qwen3 8B | Quality 7, good structured output |
| 4GB | Gemma 3 4B or Phi-4 Mini | Both achieve rank > 0.89 |

### Embedding Models (Semantic Duplicate Detection)

Used by the world generation pipeline to detect semantic duplicates among entity names (e.g., "Shadow Council" vs "Council of Shadows"). Selected via the `embedding_model` setting, not the agent auto-selection system.

#### BGE-M3 (Recommended)

- **Parameters:** 567M
- **Dimensions:** 1024
- **Strengths:** Top-ranked multilingual embedding model, excellent short-text differentiation
- **VRAM:** ~2 GB
- **Ollama:** `bge-m3`

#### MxBAI Embed Large

- **Parameters:** 335M
- **Dimensions:** 1024
- **Strengths:** High-quality embeddings, fast inference, good balance of quality and speed
- **VRAM:** ~1 GB
- **Ollama:** `mxbai-embed-large`

#### Snowflake Arctic Embed 335M

- **Parameters:** 335M
- **Dimensions:** 1024
- **Strengths:** Top-ranked on MTEB, strong retrieval performance
- **VRAM:** ~1 GB
- **Ollama:** `snowflake-arctic-embed:335m`

> **Note:** Story Factory includes automatic fallback — if the configured embedding model fails a sanity check (returns degenerate embeddings for obviously different names), it tries other installed embedding models before disabling semantic duplicate detection.

---

## HuggingFace Models (Manual Install)

Some top models aren't on Ollama and require manual download:

### DeepSeek-R1-Distill-Qwen-14B

```bash
# Download GGUF
wget https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B.Q4_K_M.gguf

# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./DeepSeek-R1-Distill-Qwen-14B.Q4_K_M.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
EOF

# Import to Ollama
ollama create deepseek-r1-14b -f Modelfile
```

### Lyra-Gutenberg 12B

```bash
# Download GGUF from HuggingFace
# https://huggingface.co/nbeerbower/Lyra-Gutenberg-mistral-nemo-12B

# Use Mistral instruct template (NOT ChatML)
cat > Modelfile << 'EOF'
FROM ./lyra-gutenberg-12b.Q5_K_M.gguf
TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
EOF
```

### Dark Champion V2 21B

```bash
# Download from HuggingFace
# https://huggingface.co/DavidAU/Llama-3.2-8X4B-MOE-V2-Dark-Champion-Instruct-uncensored-abliterated-21B-GGUF

# Use Llama 3 instruct template
```

---

## Quantization Guide

### RTX 4090 Decision Matrix

| Model Size | Recommended Quant | VRAM Usage | Quality Retention |
|------------|------------------|------------|-------------------|
| 8B | Q6_K or Q8_0 | 6.6-8.5GB | ~98-99% |
| 14B | Q5_K_M | ~9.5GB | ~95-97% |
| 30-32B | Q4_K_M | ~18-20GB | ~92-95% |
| 70B | IQ3_M + offload | ~26GB + RAM | ~85-90% |

**For creative writing:** Community consensus recommends **Q5_K_M as minimum** for nuanced prose, with Q6_K or Q8_0 preferred when VRAM permits.

### Context Window Impact

The KV cache grows linearly with context length:

| Context | KV Cache (14B) | Total VRAM (Q4_K_M) |
|---------|---------------|---------------------|
| 8K | ~2GB | ~10.5GB |
| 16K | ~4GB | ~12.5GB |
| 32K | ~8GB | ~16.5GB |
| 64K | ~16GB | ~24GB (limit) |

**RTX 4090 Sweet Spots:**
- 32B model @ Q4_K_M with 8-16K context
- 14B model @ Q5_K_M with 32K context
- 8B model @ Q4_K_M with 128K context

### IQ vs K Quantization

- **K-quants (Q4_K_M):** Default choice - fast, reliable, universal compatibility
- **I-quants (IQ4_XS):** ~10% smaller at similar quality, slower CPU inference

For multi-agent systems with CUDA: IQ4_XS provides worthwhile space savings when running multiple models simultaneously.

---

## Research Findings

### Writer Role: What Makes Good Fiction Models?

Research from [EQ-Bench Creative Writing Benchmark](https://eqbench.com/creative_writing.html) and [Lech Mazur's Writing Benchmark V4](https://github.com/lechmazur/writing) shows:

**Key evaluation criteria:**
- Character depth and motivation
- Plot structure and coherence
- World building and atmosphere
- Originality and thematic cohesion
- Line-level prose quality
- Avoidance of "GPT-isms" (overused phrases)

**Finding:** Models fine-tuned specifically on fiction (Gutenberg, Writing Prompts) outperform general-purpose models for prose quality.

### Architect Role: Narrative Planning

From [arxiv research on LLM narrative planning](https://arxiv.org/abs/2506.10161):

> "LLMs have some ability to solve planning problems but tend to break down on larger or more complicated problems."

> "GPT o1 model can solve narrative planning with ~50% accuracy for complex scenarios."

**Finding:** Reasoning-focused models (Qwen3-30B-A3B, DeepSeek-R1) significantly outperform creative models for structure planning. The hybrid thinking mode in Qwen3 is particularly valuable for this.

### Continuity Role: The Hardest Task

From [FlawedFictions benchmark](https://arxiv.org/html/2504.11900v2):

> "Even the best-performing model (o1) obtains a score of only 0.53 on long-form plot hole detection, barely outperforming random baseline."

> "LLM story generation introduces 100%+ more plot holes than human writing."

**Finding:** This is genuinely difficult. Use your strongest reasoning model, consider multiple passes, and process stories in chunks rather than full-length.

### Judge Role: Example Score Copying (Issue #228)

Empirical testing of 16 local models revealed that **hardcoded numeric examples in
JSON output format instructions cause models to copy those exact values** instead of
independently evaluating content. This affects all model sizes from 0.6B to 18B.

The solution: replace example values like `"coherence": 6.7` with parametric
placeholders `"coherence": <float 0-10>`. This single change:
- Eliminated copying (65-100% copy rate → 0-8%) across all models
- Improved MAE by 0.1-1.0 points for most models
- Increased rank correlation from 0.36-0.64 to 0.78-0.99 for capable models
- Increased score spread (variety across quality tiers) by 3-7x

**Implication for LLM prompt design:** Never put concrete numeric values in JSON
output format examples when you want the model to produce its own assessment.
Use type placeholders instead.

Full benchmark data: `scripts/evaluate_judge_accuracy.py`, results in
`output/diagnostics/judge_accuracy_*.json`.

### Editor Role: Voice Consistency

**Finding:** Using different models for writer and editor causes voice inconsistency. Consider:
1. Same model as writer with lower temperature (0.5-0.6)
2. Or dedicated editing model but accept some voice drift

### Abliteration vs Fine-Tuning

From [uncensored LLM research](https://www.watsonout.com/editorials/the-sovereign-stack-best-uncensored-llms-for-local-inference-dec-2025/):

> "Abliteration typically causes some intelligence loss - reduced reasoning, increased hallucinations."

**Finding:** Abliterated models are more compliant but may have slightly reduced capability. Fine-tuned uncensored models (Dolphin, Celeste) often preserve more intelligence.

### Multi-Agent Architecture (Agents' Room Framework)

From [ICLR 2025 research](https://arxiv.org/abs/2410.02603):

The Agents' Room framework decomposes story writing into:
- **Planning agents:** CONFLICT, CHARACTER, SETTING, PLOT (structure without prose)
- **Writing agents:** EXPOSITION, RISING ACTION, CLIMAX, FALLING ACTION, RESOLUTION

Key features:
- Scratchpad memory for shared context
- Centralized orchestrator
- Deterministic flow: planning → writing

---

## Adaptive Learning & Recommendations

Story Factory includes an adaptive learning system that tracks generation performance and suggests model/temperature adjustments.

### How It Works

1. **Generation Tracking**: Each chapter generation records:
   - Model used, mode settings, genre
   - Tokens generated, time taken, tokens/second
   - Chapter and project IDs for context

2. **Implicit Signals**: The system tracks user actions:
   - **Regenerate**: Indicates dissatisfaction (negative signal)
   - **Edit**: Measures edit distance from original
   - **Rating**: Explicit 1-5 star rating

3. **Tuning Triggers**: Recommendations are generated based on:
   - `off`: No automatic analysis
   - `after_project`: After story completion
   - `periodic`: After N chapters (e.g., every 10 chapters)
   - `continuous`: Background analysis after each generation

4. **Recommendation Types**:
   | Type | Description |
   |------|-------------|
   | `model_swap` | Suggests switching to a better-performing model |
   | `temp_adjust` | Suggests temperature change for an agent role |
   | `mode_change` | Suggests a different generation mode |
   | `vram_strategy` | Suggests VRAM strategy adjustment |

5. **Autonomy Levels**:
   - `manual`: All changes require approval
   - `cautious`: Auto-apply temp changes, prompt for model swaps
   - `balanced`: Auto-apply when confidence > threshold
   - `aggressive`: Auto-apply all, just notify
   - `experimental`: Try variations to gather data

### Settings UI

Configure learning in Settings → Learning:
- **Autonomy Level**: How much control the system has
- **Learning Triggers**: When to generate recommendations
- **Learning Threshold**: Confidence threshold for recommendations

### Recommendation Dialog

When recommendations are generated, a dialog shows:
- Recommendation type and affected role
- Current value → Suggested value
- Confidence percentage
- Expected improvement

Users can select which recommendations to apply or dismiss them.

### Disabling Learning

To completely opt out of the learning system:

1. Go to **Settings → Learning**
2. Set **Learning Triggers** to `Off`
3. Alternatively, set **Autonomy Level** to `suggest_only` to see recommendations without auto-applying

With triggers set to `Off`, no generation data is analyzed and no recommendations are generated.

### Example Scenario

After writing 10 chapters with `dolphin3:8b` for the Writer role:
- 3 chapters were regenerated (negative signal)
- Average edit distance was high (200+ characters)
- `qwen2.5:32b` showed 15% higher prose quality in other projects

The system generates a recommendation:
> **Model Swap** for Writer role
> Current: `dolphin3:8b` → Suggested: `qwen2.5:32b`
> Confidence: 78% | Expected: +15% quality

You can approve to switch models or dismiss to keep current settings.

---

## Configuration Examples

### settings.json for 24GB VRAM (2026 Update)

```json
{
  "default_model": "huihui_ai/dolphin3-abliterated:8b",
  "use_per_agent_models": true,
  "agent_models": {
    "writer": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
    "architect": "huihui_ai/qwen3-abliterated:30b",
    "editor": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
    "continuity": "deepseek-r1-14b",
    "interviewer": "huihui_ai/dolphin3-abliterated:8b",
  }
}
```

### Temperature Guidelines

| Role | Temperature | Why |
|------|-------------|-----|
| Writer | 0.7-1.0 | Creative, varied prose |
| Architect | 0.2-0.4 | Consistent logical structure |
| Editor | 0.5-0.6 | Balanced refinement |
| Continuity | 0.0-0.2 | Deterministic consistency checking |
| Interviewer | 0.4-0.6 | Balanced exploration |

**Note:** Higher temperatures correlate weakly with novelty but moderately with incoherence. They increase variation risk without guaranteeing better creativity.

---

## Troubleshooting

### Chinese characters in output
- **Cause:** Qwen3 abliterated (v1) leaks Chinese "chain of thought"
- **Fix:** Use Qwen3 v2 with layer-0 fix, or switch to Dolphin/Llama

### Model runs out of VRAM
- Use more aggressive quantization (Q4_K_M instead of Q8)
- Close other GPU applications
- Try `ollama run model --verbose` to see memory usage
- Consider sequential mode (unload models between agents)

### Poor creative writing quality
- Try Celeste, Lyra-Gutenberg, or Midnight-Miqu
- Increase temperature to 0.9-1.0
- Use longer context for better coherence
- Check that you're using Q5_K_M or better quantization

### Inconsistent story voice
- Use same model for writer and editor
- Or accept voice drift with different models

### Slow generation with big models
- Consider Qwen3-30B-A3B (MoE) instead of dense 70B
- Use IQ4_XS quantization for parallel model loading
- Reduce context window to 8-16K if not needed

---

## Model Creators to Follow

| Creator | Specialty | HuggingFace |
|---------|-----------|-------------|
| **DavidAU** | MOE models, brainstorm modifications, 355+ creative models | huggingface.co/DavidAU |
| **nbeerbower** | Gutenberg literature fine-tunes | huggingface.co/nbeerbower |
| **nothingiisreal** | Celeste series, Reddit-trained creative | huggingface.co/nothingiisreal |
| **Sao10K** | Lyra series, high-quality RP | huggingface.co/Sao10K |
| **sophosympatheia** | Midnight-Miqu, Midnight-Rose | huggingface.co/sophosympatheia |
| **huihui-ai** | Abliterated model collection | huggingface.co/huihui-ai |
| **bartowski** | High-quality GGUF quantizations | huggingface.co/bartowski |

---

## Sources

### Benchmarks & Leaderboards
- [EQ-Bench Creative Writing v3](https://eqbench.com/creative_writing.html) - LLM-judged creative writing benchmark
- [Lech Mazur Writing Benchmark V4](https://github.com/lechmazur/writing) - Tests story element incorporation
- [EQ-Bench Longform Creative Writing](https://eqbench.com/creative_writing_longform.html) - Long-form evaluation
- [Writing Styles Study](https://github.com/lechmazur/writing_styles) - Style diversity analysis

### Research Papers
- [Can LLMs Generate Good Stories? (arxiv)](https://arxiv.org/abs/2506.10161) - Narrative planning perspective
- [FlawedFictions: Plot Hole Detection (arxiv)](https://arxiv.org/html/2504.11900v2) - Consistency evaluation
- [Agents' Room (ICLR 2025)](https://arxiv.org/abs/2410.02603) - Multi-agent story generation
- [Survey on LLMs for Story Generation (ACL)](https://aclanthology.org/2025.findings-emnlp.750.pdf)

### Model Sources
- [Qwen3-30B-A3B Abliterated](https://huggingface.co/huihui-ai/Qwen3-30B-A3B-abliterated)
- [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-GGUF)
- [Celeste V1.9 on HuggingFace](https://huggingface.co/nothingiisreal/MN-12B-Celeste-V1.9)
- [Lyra-Gutenberg on HuggingFace](https://huggingface.co/nbeerbower/Lyra-Gutenberg-mistral-nemo-12B)
- [Dark Champion V2 Collection](https://huggingface.co/collections/DavidAU/dark-champion-collection-moe-mixture-of-experts)
- [Midnight-Miqu on HuggingFace](https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5)
- [SmolLM2 on HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF)
- [Dolphin 3.0 on Ollama](https://ollama.com/huihui_ai/dolphin3-abliterated)

### Community Resources
- [LocalLLaMA Creative Writing Discussion](https://huggingface.co/datasets/John6666/forum3/blob/main/creative_writing_reasoning_switch_llm_1.md)
- [Best Uncensored LLMs Guide](https://www.arsturn.com/blog/finding-the-best-uncensored-llm-on-ollama-a-deep-dive-guide)
- [Sovereign Stack: Uncensored LLMs Dec 2025](https://www.watsonout.com/editorials/the-sovereign-stack-best-uncensored-llms-for-local-inference-dec-2025/)

### Model Providers
- [Ollama Model Library](https://ollama.com/library)
- [HuggingFace Models](https://huggingface.co/models)
- [OpenRouter (for API access)](https://openrouter.ai/)
