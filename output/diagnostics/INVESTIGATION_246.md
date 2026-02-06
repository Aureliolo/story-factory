# Issue #246 Investigation: Refinement Loop Inefficiencies

## Date: 2026-02-06

## Status: Phase 3 Complete (Implementation) — see PR #261

---

## Environment

- **GPU**: NVIDIA RTX 4090 (22.6 GiB VRAM)
- **Creator model**: vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0
- **Judge model**: huihui_ai/qwen3-abliterated:30b
- **Current config**: threshold=8.0, max_iterations=3, early_stop_min_iter=3, judge_temp=0.1, creator_temp=0.9, judge_multi_call=2

## Phase 1: Diagnostic Results

### Step 1a: Judge Consistency (evaluate_judge_consistency.py)

Tested character and relationship judges with 5 repeated calls on the same frozen entity.

| Type         | Verdict    | Mean | Std  | Range   | Feedback Similarity |
|--------------|------------|------|------|---------|---------------------|
| character    | CONSISTENT | 7.4  | 0.00 | 7.4-7.4 | 0.74                |
| relationship | CONSISTENT | 7.4  | 0.01 | 7.4-7.4 | 0.62                |

**Conclusion**: Judge is essentially deterministic at temp=0.1. Not the problem.

Output: `output/diagnostics/judge_consistency_20260206_061103.json`

### Step 1b: Refinement Loop (evaluate_refinement.py)

Ran full create-judge-refine loop for character, faction, and relationship (2 entities each).

| Type         | Pass% | Plateau% | Diff Ratio | Avg Improvement/iter | Notes                                |
|--------------|-------|----------|------------|----------------------|--------------------------------------|
| character    | 50%   | **100%** | **0.000**  | 0.000                | Zephyrine: 6.5x3 (echoed). Soren: passed at 8.0 on create |
| faction      | **0%**| **100%** | 0.024      | 0.000                | Both stuck at 7.5x3                  |
| relationship | 50%   | 66.7%    | 0.259      | 0.217                | One echoed (7.4x3), one improved (7.4->8.1) |

**Conclusion**: Refiner echoes input most of the time (diff=0.0). When it does make changes (diff>0.5), scores can jump above threshold. Scores cluster at 7.4-7.5.

Output: `output/diagnostics/refinement_20260206_061226.json`

### Step 1c: A/B Prompt Test (evaluate_ab_prompts.py)

Compared current production refine prompts (A) vs improved prompts with per-dimension scores + actionable instructions (B).

| Type      | Baseline | Delta A (current) | Delta B (improved) | A-B Delta | B Wins |
|-----------|----------|--------------------|---------------------|-----------|--------|
| character | 7.6      | -0.11 (worse!)     | +0.16               | **+0.27** | 2/2    |
| faction   | 7.5      | +0.06              | +0.36               | **+0.30** | 1/2    |

**Conclusion**: Improved prompts help (+0.27 to +0.30), but still not enough to reliably reach 8.0. Current prompts sometimes make characters worse.

Output: `output/diagnostics/ab_prompts_20260206_061550.json`

---

## Root Causes Identified

### RC1: Unreachable Threshold (8.0)

Typical creation scores land at 7.4-7.5. Even improved prompts only get ~50% of entities to 8.0. The threshold asks for "excellent" quality from a 12b creator model.

### RC2: Calibration Block Suppresses High Scores

The judge calibration block (`_common.py:20-32`) creates a psychological ceiling:

```text
- 8-9: Excellent (genuinely impressive, few weaknesses — justify in feedback)
```

Rule 3: "If you give 8+ on a dimension, your feedback MUST explain what makes it exceptional."

This creates a brake - the model avoids 8+ because it requires self-justification. At temp=0.1, the judge locks into the 7.4-7.5 "safe zone" between "competent" (6-7) and "strong" (7-8).

Additionally, the formatting examples in Rule 1 (`e.g., 5.3, 7.1, 8.6`) may subtly anchor score range perception.

### RC3: Refiner Echoes Input

The current production refine prompts don't tell the model WHAT to improve. Without per-dimension scores and actionable instructions, the creator model returns identical or near-identical content (diff_ratio=0.0).

### RC4: Early Stopping is Disabled

`early_stopping_min_iterations=3` equals `max_iterations=3`, meaning early stopping can never trigger. When diff=0.0, two extra API calls are wasted per entity.

### RC5: No Unchanged-Output Detection

Even if the refiner returns identical content, the system re-judges it (getting the same score) and continues to the next iteration. No early exit when output hasn't changed.

---

## Phase 2: Planned Variant Testing

### Goal

Empirically determine which combination of calibration changes + threshold unlocks higher scores without losing score quality (differentiation, accuracy).

### Variants to Test

We need a standalone script that tests different judge calibration variants against the same frozen entities, measuring:
- Score range and distribution
- Differentiation between dimensions
- Whether the judge can distinguish good vs mediocre entities
- Whether scores become inflated/meaningless

#### Variant A: Current Production (Baseline)
- Full calibration block with 8+ justification rule
- `<float 0-10>` parametric output format
- This is what we already measured: scores cluster at 7.4-7.5

#### Variant B: Softened 8+ Rule
- Same calibration block but change Rule 3 from:
  "If you give 8+ on a dimension, your feedback MUST explain what makes it exceptional."
  To:
  "Use 8+ scores when deserved — strong work should score in the 7-9 range."
- Hypothesis: removes the brake, allows scores to spread into 8+ territory

#### Variant C: Removed Calibration Block Entirely
- No SCORING GUIDE, no RULES
- Just the entity, dimensions, and output format
- Hypothesis: scores may inflate but will test the natural scoring range

#### Variant D: Simplified Calibration (No Ranges, No Rules)
- Keep dimension descriptions but remove the 1-10 range labels and rules
- Just: "Score each dimension 0-10 with one decimal place."
- Hypothesis: less anchoring, more natural spread

#### Variant E: Flattened Calibration (No Tiers)
- Remove the tier labels ("Competent", "Strong", "Excellent") but keep the decimal/differentiation rules
- Hypothesis: tier labels anchor scores, removing them unblocks 8+

#### Variant F: Encouraging Calibration
- Same structure but reword to encourage higher scores for good work:
  "7-9: Most well-crafted entities should score in this range"
  "5-6: Generic or bland entities"
  "1-4: Broken or contradictory"
- Hypothesis: shifts the "comfort zone" upward

### Test Protocol

1. Create 3-4 seed entities of known quality (1 mediocre, 1 good, 1 excellent - from evaluate_judge_accuracy.py ground truth samples)
2. Judge each entity with each variant (5 calls per variant for stability)
3. Measure per-variant:
   - Mean score per entity
   - Score spread across entities (can it differentiate quality tiers?)
   - Per-dimension differentiation (std across dimensions within an entity)
   - Whether mediocre < good < excellent ordering is preserved (rank correlation)
4. Compare against ground truth from evaluate_judge_accuracy.py samples

### Success Criteria

The winning variant should:
- Allow good entities to score 7.5+ (reachable threshold)
- Maintain quality ordering: mediocre < good < excellent
- Keep dimension differentiation (not all dimensions get the same score)
- Not inflate mediocre entities above the threshold

---

## Previous Diagnostic Runs (Historical Context)

### Jan 31 Run (threshold was 7.5 at that time)

| Type         | Pass% | Plateau% | Diff Ratio | Notes |
|--------------|-------|----------|------------|-------|
| character    | 33%   | 100%     | 0.000      | Refiner echoed input                  |
| faction      | 100%  | 0%       | 0.319      | Worked well at 7.5 threshold          |
| concept      | 100%  | 0%       | 0.000      | All passed on first create            |
| item         | 100%  | 0%       | 0.000      | All passed on first create            |
| location     | 100%  | 0%       | 0.000      | All passed on first create            |
| relationship | 67%   | 75%      | 0.146      | Partial echoing                       |

Models at that time: creator=vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0, judge=huihui_ai/dolphin3-abliterated:8b

**Key difference**: Judge was 8b dolphin3, now it's 30b qwen3. Threshold was 7.5, now 8.0. Most entity types that passed at 7.5 would fail at 8.0.

### Judge Accuracy Benchmark (Feb 4)

File: `output/diagnostics/judge_accuracy_20260204_074030.json`

Tested multiple models against ground-truth samples. This provides baseline accuracy data for the judge models.

---

## Phase 2: Calibration Variant Benchmark Results

Script: `scripts/evaluate_calibration_variants.py`
Output: `output/diagnostics/calibration_variants_20260206_063601.json`
Total time: 107s (108 LLM calls)

### Test Setup

- 6 ground-truth samples: terrible/mediocre/excellent x character/faction
- 3 judge calls per sample per variant (for stability)
- Same judge model (qwen3-abliterated:30b) at temp=0.1
- Same `generate_structured()` + instructor pipeline as production

### Raw Results

| Variant          | Terrible | Mediocre | Excellent | Order | Rank Corr | Dim Spread | Gap T-M | Gap M-E | T=7.5 | T=8.0 |
|------------------|----------|----------|-----------|-------|-----------|------------|---------|---------|-------|-------|
| A_production     | 6.2      | 7.0      | 8.0       | OK    | 0.94      | 1.6        | 0.8     | 1.0     | OK    | OK    |
| B_softened       | 6.6      | 7.3      | 8.1       | OK    | 0.93      | 1.6        | 0.7     | 0.8     | OK    | OK    |
| C_no_calibration | 5.8      | 7.3      | 8.2       | OK    | **1.00**  | **2.8**    | **1.5** | 1.0     | OK    | OK    |
| D_minimal        | 5.6      | 7.2      | 8.3       | OK    | 0.94      | **2.7**    | **1.6** | 1.0     | OK    | OK    |
| E_no_tiers       | 6.5      | 7.3      | 8.0       | OK    | 0.89      | 2.2        | 0.8     | 0.7     | OK    | OK    |
| F_encouraging    | 6.5      | 7.5      | 8.3       | OK    | **0.99**  | 1.8        | 1.0     | 0.8     | FAIL  | OK    |

### Key Findings

**All variants maintain quality ordering** (terrible < mediocre < excellent). No variant inflated mediocre entities above excellent ones. The judge fundamentally differentiates quality regardless of calibration wording.

**Calibration block IS suppressing scores, but less than expected.** The production variant (A) already gets excellent entities to 8.0 average in this controlled test. The 7.4-7.5 ceiling we saw in Phase 1 was partly due to entity quality - the production entities weren't as strong as the "excellent" ground-truth samples.

**Removing calibration entirely (C, D) gives best differentiation:**
- Highest dim spread (2.7-2.8 vs 1.6 production) - the judge uses the full scale more
- Widest gap between terrible and mediocre (1.5-1.6 vs 0.8 production)
- Perfect rank correlation (C: 1.00)
- Scores use more of the range (terrible drops to 5.6-5.8)

**Encouraging calibration (F) inflates mediocre scores:**
- Mediocre entities score 7.5 = threshold, meaning bad work passes at 7.5 threshold
- Fails the 7.5 threshold test (mediocre == threshold, not safely below it)
- Works at 8.0 threshold though

**B_softened is marginal improvement over A** - barely changes anything (0.1 higher across the board). The 8+ justification rule is NOT the main brake.

**E_no_tiers has worst rank correlation** (0.89) - removing tier labels without removing rules makes the judge less discriminating, not more.

### Analysis: Which Variant Wins?

**Best overall: D_minimal** (or C_no_calibration). F_encouraging was considered but rejected because mediocre entities score exactly 7.5 at threshold=7.5, meaning bad work would pass.

| Criterion                  | D_minimal | C_no_calibration | F_encouraging |
|---------------------------|-----------|------------------|---------------|
| Excellent avg             | 8.3       | 8.2              | 8.3           |
| Mediocre safely below 7.5 | 7.2 (yes) | 7.3 (yes)        | 7.5 (no!)     |
| Rank correlation          | 0.94      | 1.00             | 0.99          |
| Dim spread                | 2.7       | 2.8              | 1.8           |
| Score consistency (std)   | Some var  | Very stable      | Very stable   |

**C_no_calibration** has the best rank correlation (perfect 1.00) and widest differentiation. But having NO calibration at all feels fragile - different models may behave very differently without any guidance.

**D_minimal** keeps minimal guidance ("Score 0-10 with one decimal, differentiate dimensions") without the oppressive tier labels and rules. Good balance of structure and freedom. Near-perfect rank correlation (0.94), highest excellent scores (8.3), and mediocre stays safely at 7.2.

### Recommendation

**Adopt variant D_minimal calibration + lower threshold to 7.5.**

Rationale:
- D_minimal excellent=8.3, easily passes 7.5
- D_minimal mediocre=7.2, safely below 7.5 (0.3 gap)
- D_minimal terrible=5.6, far below any threshold
- Good rank correlation (0.94), excellent dimension spread (2.7)
- Provides minimal guidance so the judge uses the full scale naturally
- Threshold 7.5 is realistic for the 12b creator model to achieve after refinement

**Alternative: C_no_calibration + threshold 7.5** if we want maximum differentiation and are comfortable with zero calibration text. Has perfect rank correlation and even wider spread.

---

## Phase 3: Implementation Plan

Based on Phase 1 + Phase 2 findings, the following changes are needed:

### Fix 1: Replace Calibration Block (RC2)

Replace the current `JUDGE_CALIBRATION_BLOCK` in `_common.py` with the exact benchmarked D_minimal variant:

```text
Score each dimension 0-10 with one decimal place.
Differentiate between dimensions — scores should vary based on actual quality.
```

### Fix 2: Lower Threshold to 7.5 (RC1)

Change default `quality_threshold` from 8.0 to 7.5 in settings.

### Fix 3: Add Unchanged-Output Detection (RC5)

In the quality refinement loop (`_quality_loop.py`), add a check: if the refiner returns content identical (or near-identical, diff_ratio < 0.05) to the input, skip the re-judge and exit the loop immediately. This saves 2 wasted API calls per stalled entity.

### Fix 4: Lower Early-Stopping Min Iterations (RC4)

Change `early_stopping_min_iterations` default from 2 to 1. This allows the loop to exit after the first iteration if the entity already passes threshold. Combined with Fix 3, this means:
- Entity passes on create -> 1 judge call, done (instead of 3)
- Entity stalls (refiner echoes) -> exit immediately (instead of 2 wasted iterations)

### Fix 5: Adopt Improved Refine Prompts (RC3)

The A/B test showed improved prompts with per-dimension scores and actionable instructions give +0.27 to +0.30 improvement. The current production `_refine_*` functions already have per-dimension scoring in the prompt (added during earlier work). Verify these are active.

---

## Files Modified/Created

- This document: `output/diagnostics/INVESTIGATION_246.md`
- Step 1a output: `output/diagnostics/judge_consistency_20260206_061103.json`
- Step 1b output: `output/diagnostics/refinement_20260206_061226.json`
- Step 1c output: `output/diagnostics/ab_prompts_20260206_061550.json`
- Phase 2 script: `scripts/evaluate_calibration_variants.py`
- Phase 2 output: `output/diagnostics/calibration_variants_20260206_063601.json`

## Next Steps

1. ~~Write `scripts/evaluate_calibration_variants.py` - tests variants A-F~~ DONE
2. ~~Run the calibration variant benchmark~~ DONE
3. ~~Analyze results to pick the best calibration approach~~ DONE -> D_minimal + threshold 7.5
4. ~~Implement the winning calibration + threshold + early-exit fixes (Fixes 1-5)~~ DONE (PR #261)
5. Re-run `evaluate_refinement.py` to validate improvement (post-merge)
