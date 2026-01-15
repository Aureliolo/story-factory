# Model Guide for Story Factory

This document provides comprehensive guidance on selecting the best LLM models for Story Factory's multi-agent story generation system.

**Last Updated:** January 2026
**Hardware Reference:** RTX 4090 (24GB VRAM)

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
| **Validator** | Basic output validation | Minimal capability needed | Any small model |

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
| **Validator** | SmolLM2-1.7B-Instruct | HuggingFace | 1.2GB | **NEW**: Better quality than 0.5B models |

**Key Change from 2025:** Qwen3-30B-A3B replaces Llama 3.3 70B for reasoning tasks - same quality at half the VRAM, enabling parallel model loading.

### For 16GB VRAM

| Role | Model | VRAM |
|------|-------|------|
| **Writer** | Celeste V1.9 12B or Lyra-Gutenberg 12B | 10GB |
| **Architect** | Qwen3-14B with `/think` mode | 10GB |
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
- **Feature:** `/think` and `/no_think` toggles for reasoning mode
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
- **Ollama:** `huihui_ai/llama3.3-abliterated:70b-q4_K_M`

### General Purpose Models

#### Dolphin 3.0 8B (Eric Hartford)
- **Strengths:** Highly steerable via system prompts, excellent instruction following, no refusals
- **Best for:** Interviewer role, fast tasks, coding/math
- **Personality:** "Less about personality, all about raw, unfiltered smarts"
- **Ollama:** `huihui_ai/dolphin3-abliterated:8b`

#### Dolphin Mistral Nemo 12B
- **Strengths:** 128K context window, good for editing and refinement
- **Ollama:** `CognitiveComputations/dolphin-mistral-nemo:12b`

### Tiny Models (Validator Role)

#### SmolLM2-1.7B-Instruct (NEW - Recommended)
- **Strengths:** Best quality-per-VRAM ratio for validation tasks
- **VRAM:** ~1.2GB
- **HuggingFace:** `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF`

#### Qwen3-0.6B
- **Strengths:** Ultra-fast validation, minimal resource usage
- **VRAM:** ~0.5GB
- **Ollama:** `qwen3:0.6b`

#### SmolLM2-360M-Instruct
- **Strengths:** Near-instant validation passes
- **VRAM:** ~0.3GB
- **Best for:** High-volume validation where speed matters most

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

### SmolLM2-1.7B-Instruct

```bash
# Download from HuggingFace
# https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF

cat > Modelfile << 'EOF'
FROM ./smollm2-1.7b-instruct.Q4_K_M.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOF

ollama create smollm2:1.7b -f Modelfile
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
    "validator": "smollm2:1.7b"
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
| Validator | 0.1 | Minimal variation |

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
