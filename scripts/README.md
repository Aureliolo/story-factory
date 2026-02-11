# Scripts

Standalone utility scripts for development, diagnostics, and operations. These are **not** part of the main application and are excluded from test coverage requirements.

## Quick Reference

| Script                            | Purpose                          | When to use                                     |
| --------------------------------- | -------------------------------- | ----------------------------------------------- |
| `evaluate_judge_consistency.py`   | Measure judge scoring variance   | First step when diagnosing refinement failures  |
| `evaluate_refinement.py`          | Full refinement loop instrumentation | Main diagnostic for refinement loop issues  |
| `evaluate_ab_prompts.py`          | A/B test prompt variants         | After identifying prompt-related problems       |
| `control_panel.py`                | Desktop GUI for app management   | Running Story Factory on desktop                |
| `healthcheck.py`                  | Verify environment setup         | After install or when things break              |
| `check_deps.py`                   | Check/install dependencies       | After updating pyproject.toml                   |
| `check_file_size.py`              | Enforce max file length          | Pre-commit hook (automatic)                     |
| `audit_exceptions.py`             | Audit exception handling patterns | Code quality review                            |

---

## Diagnostic Scripts (Issue #223)

Three scripts for investigating world quality refinement loop failures. They instrument the create-judge-refine pipeline to pinpoint whether problems originate in the judge, the refiner, or the prompts.

**Prerequisites:**
- Ollama running locally with a model pulled (e.g. `huihui_ai/dolphin3-abliterated:8b`)
- `src/settings.json` exists with valid config (copy from `src/settings.example.json`)

**Execution order matters** -- run them in the order listed below. Each script's results inform whether the next script is needed.

### Step 1: `evaluate_judge_consistency.py`

Measures judge scoring variance by calling the judge N times on the **same frozen entity**. Determines if the judge is noisy (unreliable) or consistent.

```bash
python scripts/evaluate_judge_consistency.py [options]
  --entity-types faction,concept  # Comma-separated (default: all 6)
  --judge-calls 5                 # Calls per entity (default: 5)
  --output results.json           # Output path (default: output/diagnostics/<timestamp>_judge.json)
  --verbose                       # Print per-call scores
```

**Example (quick smoke test):**

```bash
python scripts/evaluate_judge_consistency.py --entity-types faction --judge-calls 3 --verbose
```

**Output interpretation:**

| Verdict                    | Meaning                | Next step                                                                    |
| -------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| `CONSISTENT` (std < 0.2)  | Judge is reliable      | Proceed to Step 2                                                            |
| `BORDERLINE` (std 0.2-0.5) | Some noise, not critical | May benefit from multi-call averaging                                      |
| `NOISY` (std > 0.5)       | Judge is unreliable    | Fix judge first (lower temperature, structured output, multi-call averaging) |

Key fields in the JSON output:
- `results[].per_dimension_stats` -- std per scoring dimension (the key metric)
- `results[].verdict` -- per-entity-type verdict
- `results[].feedback_similarity` -- Jaccard overlap between feedback strings (low = inconsistent text)

### Step 2: `evaluate_refinement.py`

Runs the full create-judge-refine loop with instrumentation, capturing entity snapshots, scores, feedback, and timing at every iteration.

```bash
python scripts/evaluate_refinement.py [options]
  --entity-types faction,concept  # Comma-separated (default: all 6)
  --count-per-type 3              # Entities per type (default: 3)
  --output results.json           # Output path (default: output/diagnostics/<timestamp>.json)
  --verbose                       # Print per-iteration scores
```

**Example (quick smoke test):**

```bash
python scripts/evaluate_refinement.py --entity-types faction --count-per-type 1 --verbose
```

**Output interpretation:**

| Metric                  | Where in JSON                                  | Healthy    | Broken           |
| ----------------------- | ---------------------------------------------- | ---------- | ---------------- |
| Pass rate               | `summary.pass_rate_by_type`                    | >50%       | <30%             |
| Score improvement/iter  | `summary.avg_score_improvement_per_iteration`  | >0.3       | <0.1             |
| Description diff ratio  | `summary.avg_description_diff_ratio`           | 0.15-0.40  | <0.10 or >0.50   |
| Plateau rate            | `summary.plateau_rate`                         | <30%       | >60%             |
| Regression rate         | `summary.regression_rate`                      | <15%       | >30%             |
| Feedback specificity    | `summary.avg_feedback_specificity`             | >0.15      | <0.05            |

**Decision tree:**

1. **diff_ratio < 0.10** -- Refiner makes cosmetic changes only. Fix: improve refine prompts.
2. **diff_ratio > 0.30 but score improvement < 0.2** -- Refiner changes substance but judge doesn't reward it. Fix: differentiate judge prompt.
3. **Feedback specificity < 0.05** -- Judge gives generic feedback the refiner can't act on. Fix: improve judge prompt.
4. **Specificity > 0.10 but no improvement** -- Good feedback, refiner ignores it. Fix: improve refine prompts to use feedback.
5. **Plateau rate > 60%** -- Iterations wasted, scores don't move. Likely cause: threshold/calibration conflict.
6. **Regression rate > 30%** -- Refinement makes things worse. Likely cause: temperature decay too aggressive.

### Step 3: `evaluate_ab_prompts.py`

A/B tests current production prompts (A) vs improved variants (B) on the same seed entities. Run this if Step 2 points to prompt problems.

```bash
python scripts/evaluate_ab_prompts.py [options]
  --entity-types faction,concept  # Comma-separated (default: character,faction,concept,item,location)
  --count-per-type 3              # Seeds per type (default: 3)
  --output results.json           # Output path (default: output/diagnostics/<timestamp>_ab.json)
  --verbose                       # Print per-entity scores
```

**What the improved prompts change:**
- Use actual quality threshold from config instead of hardcoded "need 9+"
- Include per-dimension numeric scores
- Add actionable improvement instructions derived from low-scoring dimensions
- Include judge feedback more prominently

**Output interpretation:**

| `avg_delta_ab` | Meaning                            | Action                                  |
| -------------- | ---------------------------------- | --------------------------------------- |
| > 0.5          | Improved prompts help significantly | Implement prompt changes               |
| 0.2 - 0.5     | Modest improvement                 | Consider implementing                   |
| -0.2 - 0.2    | No significant difference          | Prompt changes alone won't fix this     |
| < -0.2         | Current prompts are better         | Investigate other causes                |

### Diagnostic Output Files

All output goes to `output/diagnostics/`:
- `judge_consistency_<timestamp>.json`
- `refinement_<timestamp>.json`
- `ab_prompts_<timestamp>.json`

---

## Operations Scripts

### `control_panel.py`

Native desktop GUI (CustomTkinter) for managing the Story Factory application. Start/stop the server, monitor logs, open the browser.

```bash
python scripts/control_panel.py
```

Requires `customtkinter` (optional dependency, not installed by default).

### `healthcheck.py`

Verifies the environment is correctly set up: Python version, required packages, Ollama connectivity, output directories.

```bash
python scripts/healthcheck.py
```

### `check_deps.py`

Compares installed package versions against `pyproject.toml` requirements. Optionally auto-installs missing or outdated packages.

```bash
python scripts/check_deps.py              # Check only
python scripts/check_deps.py --auto-install  # Check and fix
```

---

## Development Scripts

### `check_file_size.py`

Pre-commit hook that enforces the 1000-line maximum for Python files in `src/`. Runs automatically via pre-commit; not typically invoked manually.

```bash
python scripts/check_file_size.py [files...]
```

### `audit_exceptions.py`

Scans the codebase for exception handling patterns. Reports broad `except Exception` catches, missing logging in handlers, and missing `from` clauses on re-raises.

```bash
python scripts/audit_exceptions.py
```
