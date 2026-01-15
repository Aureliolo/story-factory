# Model Guide for Story Factory

This document provides comprehensive guidance on selecting the best LLM models for Story Factory's multi-agent story generation system.

**Last Updated:** January 2025
**Hardware Reference:** RTX 4090 (24GB VRAM)

## Table of Contents

- [Quick Start](#quick-start)
- [Agent Roles & Requirements](#agent-roles--requirements)
- [Recommended Model Stack](#recommended-model-stack)
- [Model Deep Dive](#model-deep-dive)
- [HuggingFace Models](#huggingface-models-manual-install)
- [Research Findings](#research-findings)
- [Sources](#sources)

---

## Quick Start

```bash
# Essential models for 24GB VRAM setup
ollama pull huihui_ai/dolphin3-abliterated:8b      # Fast default, interviewer
ollama pull vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0  # Creative writing
ollama pull huihui_ai/llama3.3-abliterated:70b-q4_K_M  # Reasoning (architect/continuity)
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

### For 24GB VRAM (RTX 4090)

| Role | Model | Source | VRAM | Why This Model |
|------|-------|--------|------|----------------|
| **Writer** | Celeste V1.9 12B | Ollama | 14GB | Purpose-built for fiction, OOC steering, excellent NSFW |
| **Writer (Premium)** | Midnight-Miqu 70B Q4 | Ollama | 24GB | "Writes like a novelist", 32K context |
| **Architect** | Llama 3.3 70B Q4_K_M | Ollama | 24GB | Best reasoning for structure planning |
| **Editor** | Same as Writer (temp 0.5) | - | - | Maintains voice consistency |
| **Continuity** | Llama 3.3 70B Q4_K_M | Ollama | 24GB | Needs reasoning, not creativity |
| **Interviewer** | Dolphin 3.0 8B | Ollama | 8GB | Fast, compliant, highly steerable |
| **Validator** | qwen2.5:0.5b | Ollama | 1GB | Basic sanity checks only |

### For 16GB VRAM

| Role | Model | VRAM |
|------|-------|------|
| **Writer** | Celeste V1.9 12B or Nemo-Gutenberg 12B | 10GB |
| **Architect** | Qwen3 14B with `/think` mode | 12GB |
| **Continuity** | Same as Architect | 12GB |
| **Interviewer** | Dolphin 3.0 8B | 8GB |

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

#### Nemo-Gutenberg 12B v2
- **Base:** Mistral NeMo 12B
- **Training:** Project Gutenberg fiction datasets (DPO)
- **Strengths:** "Best of all Nemo finetunes" per AI judges, literary prose style
- **Note:** Use Mistral chat format, NOT ChatML
- **Source:** HuggingFace (manual import required)

#### Dark Champion 18B MOE
- **Architecture:** 8x3B Mixture of Experts (18.4B total)
- **Strengths:** "OFF THE SCALE" prose quality, 50+ tok/s on 16GB, exceptional fiction
- **Cons:** MOE can be inconsistent, occasional verbosity
- **Ollama:** `TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit`

#### Midnight-Miqu 70B (Premium)
- **Base:** Miqu (Mistral-based)
- **Strengths:** "Writes like a novelist", 32K context, emotional depth + narrative coherence
- **VRAM:** 24GB+ at Q4 quantization
- **Ollama:** `vanilj/midnight-miqu-70b-v1.5`

#### Gemma The Writer 10B
- **Base:** Gemma 2
- **Training:** Merge of top storytelling models + Brainstorm 5x adapter
- **Strengths:** Reduced GPT-isms, varied prose, 2x longer outputs than base
- **Source:** HuggingFace (`DavidAU/Gemma-The-Writer-J.GutenBerg-10B-GGUF`)

### Reasoning Models (Architect/Continuity Roles)

#### Llama 3.3 70B Abliterated
- **Strengths:** Best reasoning in uncensored local models, excellent for structure
- **Use for:** Story architecture, chapter planning, continuity checking
- **Ollama:** `huihui_ai/llama3.3-abliterated:70b` (full) or `:70b-q4_K_M` (quantized)

#### Qwen3 with Thinking Mode
- **Feature:** `/think` and `/no_think` toggles for reasoning mode
- **Strengths:** Explicit reasoning for planning tasks
- **Warning:** May output Chinese characters (abliterated v1 issue)
- **Alternative:** Use v2 versions with layer-0 fix

### General Purpose Models

#### Dolphin 3.0 8B (Eric Hartford)
- **Strengths:** Highly steerable via system prompts, excellent instruction following, no refusals
- **Best for:** Interviewer role, fast tasks, coding/math
- **Personality:** "Less about personality, all about raw, unfiltered smarts"
- **Ollama:** `huihui_ai/dolphin3-abliterated:8b`

#### Dolphin Mistral Nemo 12B
- **Strengths:** 128K context window, good for editing and refinement
- **Ollama:** `CognitiveComputations/dolphin-mistral-nemo:12b`

---

## HuggingFace Models (Manual Install)

Some of the best creative writing models aren't on Ollama and require manual download:

### Nemo-Gutenberg 12B v2

```bash
# Download GGUF
wget https://huggingface.co/mradermacher/nemo-gutenberg-12b-v2-GGUF/resolve/main/nemo-gutenberg-12b-v2.Q5_K_M.gguf

# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./nemo-gutenberg-12b-v2.Q5_K_M.gguf
TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
EOF

# Import to Ollama
ollama create nemo-gutenberg:12b -f Modelfile
```

### Gemma The Writer J.GutenBerg 10B

```bash
# Download from HuggingFace
# https://huggingface.co/DavidAU/Gemma-The-Writer-J.GutenBerg-10B-GGUF

# Use GEMMA instruct template
# Recommended: temp 0-5, rep_pen 1.05+
```

### Midnight-Miqu 70B (if not on Ollama)

```bash
# Available via vanilj on Ollama, or download GGUF:
# https://huggingface.co/mav23/Midnight-Miqu-70B-v1.5-GGUF
```

---

## Research Findings

### Writer Role: What Makes Good Fiction Models?

Research from [EQ-Bench Creative Writing Benchmark](https://eqbench.com/creative_writing.html) and [Lech Mazur's Writing Benchmark](https://github.com/lechmazur/writing) shows:

**Key evaluation criteria:**
- Character depth and motivation
- Plot structure and coherence
- World building and atmosphere
- Originality and thematic cohesion
- Line-level prose quality
- Avoidance of "GPT-isms" (overused phrases)

**Finding:** Models fine-tuned specifically on fiction (Gutenberg, Writing Prompts) outperform general-purpose models for prose quality.

### Architect Role: Narrative Planning Challenges

From [arxiv research on LLM narrative planning](https://arxiv.org/abs/2506.10161):

> "LLMs have some ability to solve planning problems but tend to break down on larger or more complicated problems."

> "GPT o1 model can solve narrative planning with ~50% accuracy for complex scenarios. Claude-3.5 has similar performance."

**Finding:** Reasoning-focused models (Llama 3.3 70B, o1-class) significantly outperform creative models for structure planning.

### Continuity Role: The Hardest Task

From [FlawedFictions benchmark](https://arxiv.org/html/2504.11900v2):

> "Even the best-performing model (o1) obtains a score of only 0.53 on long-form plot hole detection, barely outperforming random baseline."

> "Contemporary LLMs have substantial gaps in capabilities to reliably detect consistency issues in long-form narratives."

**Finding:** This is genuinely difficult. Use your strongest reasoning model, consider multiple passes.

### Editor Role: Voice Consistency Matters

**Finding:** Using different models for writer and editor can cause voice inconsistency. Consider:
1. Same model as writer with lower temperature (0.5-0.6)
2. Or dedicated editing model but accept some voice drift

### Abliteration vs Fine-Tuning

From [uncensored LLM research](https://www.watsonout.com/editorials/the-sovereign-stack-best-uncensored-llms-for-local-inference-dec-2025/):

> "Abliteration identifies and removes specific parts responsible for refusal behaviors - like a digital lobotomy to remove the 'no' button."

> "Caveat: abliteration typically causes some intelligence loss - reduced reasoning, increased hallucinations."

**Finding:** Abliterated models are more compliant but may have slightly reduced capability. Fine-tuned uncensored models (Dolphin, Celeste) often preserve more intelligence.

---

## Configuration Examples

### settings.json for 24GB VRAM

```json
{
  "default_model": "huihui_ai/dolphin3-abliterated:8b",
  "use_per_agent_models": true,
  "agent_models": {
    "writer": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
    "architect": "huihui_ai/llama3.3-abliterated:70b-q4_K_M",
    "editor": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
    "continuity": "huihui_ai/llama3.3-abliterated:70b-q4_K_M",
    "interviewer": "huihui_ai/dolphin3-abliterated:8b"
  }
}
```

### Temperature Guidelines

| Role | Temperature | Why |
|------|-------------|-----|
| Writer | 0.8-1.0 | Creative, varied prose |
| Architect | 0.3-0.5 | Logical, consistent structure |
| Editor | 0.5-0.6 | Balanced refinement |
| Continuity | 0.2-0.3 | Precise analysis |
| Interviewer | 0.7 | Natural conversation |

---

## Troubleshooting

### Chinese characters in output
- **Cause:** Qwen3 abliterated (v1) leaks Chinese "chain of thought"
- **Fix:** Use Qwen3 v2 with layer-0 fix, or switch to Dolphin/Llama

### Model runs out of VRAM
- Use more aggressive quantization (Q4_K_M instead of Q8)
- Close other GPU applications
- Try `ollama run model --verbose` to see memory usage

### Poor creative writing quality
- Try Celeste, Nemo-Gutenberg, or Midnight-Miqu
- Increase temperature to 0.9-1.0
- Use longer context for better coherence

### Inconsistent story voice
- Use same model for writer and editor
- Or accept voice drift with different models

---

## Sources

### Benchmarks & Leaderboards
- [EQ-Bench Creative Writing v3](https://eqbench.com/creative_writing.html) - LLM-judged creative writing benchmark
- [Lech Mazur Writing Benchmark](https://github.com/lechmazur/writing) - Tests story element incorporation
- [EQ-Bench Longform Creative Writing](https://eqbench.com/creative_writing_longform.html) - Long-form evaluation

### Research Papers
- [Can LLMs Generate Good Stories? (arxiv)](https://arxiv.org/abs/2506.10161) - Narrative planning perspective
- [FlawedFictions: Plot Hole Detection (arxiv)](https://arxiv.org/html/2504.11900v2) - Consistency evaluation
- [Survey on LLMs for Story Generation (ACL)](https://aclanthology.org/2025.findings-emnlp.750.pdf)

### Model Sources
- [Celeste V1.9 on HuggingFace](https://huggingface.co/nothingiisreal/MN-12B-Celeste-V1.9)
- [Nemo-Gutenberg on HuggingFace](https://huggingface.co/nbeerbower/mistral-nemo-gutenberg-12B-v2)
- [Dark Champion Collection](https://huggingface.co/collections/DavidAU/dark-champion-collection-moe-mixture-of-experts)
- [Midnight-Miqu on HuggingFace](https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5)
- [Gemma The Writer Collection](https://huggingface.co/DavidAU/Gemma-The-Writer-J.GutenBerg-10B-GGUF)
- [Dolphin 3.0 on Ollama](https://ollama.com/huihui_ai/dolphin3-abliterated)

### Community Resources
- [LocalLLaMA Creative Writing Discussion](https://huggingface.co/datasets/John6666/forum3/blob/main/creative_writing_reasoning_switch_llm_1.md)
- [Best Uncensored LLMs Guide](https://www.arsturn.com/blog/finding-the-best-uncensored-llm-on-ollama-a-deep-dive-guide)
- [Sovereign Stack: Uncensored LLMs Dec 2025](https://www.watsonout.com/editorials/the-sovereign-stack-best-uncensored-llms-for-local-inference-dec-2025/)
- [Best LLMs for Writing 2025](https://intellectualead.com/best-llm-writing/)

### Model Providers
- [Ollama Model Library](https://ollama.com/library)
- [HuggingFace Models](https://huggingface.co/models)
- [OpenRouter (for API access)](https://openrouter.ai/)
