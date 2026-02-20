---
description: "Deep log analysis: 10 parallel specialist agents + Opus coordinator synthesize findings"
argument-hint: "[log file path, or blank for default output/logs/story_factory.log]"
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
  - Task
  - Write
---

# Analyze Logs

Deep, multi-agent log analysis that finds bugs, anti-patterns, wasted work, data corruption, performance issues, and anything suspicious in Story Factory's application log.

**Arguments:** "$ARGUMENTS"

---

## Phase 1: Setup

Resolve the log file path:
- If `$ARGUMENTS` is provided and non-empty, use it as the log file path
- Otherwise, use the default: `output/logs/story_factory.log`

Validate the log file exists, then gather baseline metrics using dedicated tools:

1. Use **Bash** with `test -f <path> && echo EXISTS || echo MISSING` to verify the log file exists
2. Use **Bash** with `realpath <path>` to resolve the absolute path (use this as `<LOG_FILE>` going forward)
3. Use **Bash** with `wc -l` to get the total line count
4. Use **Grep** (output_mode: "count") for severity counts:
   - Pattern `\[ERROR\]` for errors
   - Pattern `\[WARNING\]` for warnings
   - Pattern `\[INFO\]` for info
   - Pattern `\[DEBUG\]` for debug
5. Use **Read** (limit: 1) to get the first line (session start timestamp)
6. Use **Bash** with `tail -n 1` to get the last line (session end timestamp)
7. Use **Bash** with `python -c "import datetime; now=datetime.datetime.now(); print(now.strftime('%Y-%m-%d')); print(now.strftime('%H-%M-%S'))"` to get today's date and current time

**Early exit conditions:**
- If the log file **does not exist**: report "Log file not found at `<path>`." and **stop**.
- If the log file has **0 lines**: report "Log file is empty — nothing to analyze." and **stop**.
- If the log file has **fewer than 10 lines**: report "Log file too small for multi-agent analysis — performing inline analysis." Read the entire file, analyze it directly in the main conversation (do not launch agents), present findings inline, and then **stop** (do not proceed to Phase 2).

After completing Phase 1, compute these values for substitution:
- `<LOG_FILE>` — the resolved absolute path from `realpath`
- `<LINE_COUNT>` — the total line count from `wc -l`
- `<DATE>` — today's date in YYYY-MM-DD format (first line of step 7 output)
- `<TIME>` — current time in HH-MM-SS format (second line of step 7 output)
- `<OUTPUT_PATH>` — `output/logs/LOG_ANALYSIS_<DATE>_<TIME>.md` (timestamped to avoid overwriting previous same-day analyses)

Substitute `<LOG_FILE>`, `<LINE_COUNT>`, and `<DATE>` into Phase 2 agent prompts **and** the Phase 3 coordinator prompt (which uses them in the report header). Substitute `<OUTPUT_PATH>` into Phase 4 only (the main agent writes the file; neither specialist agents nor the coordinator use it).

Print to the user:
> Analyzing **N** lines (E errors, W warnings, I info, D debug) spanning TIME_RANGE.
> Launching 10 specialist agents in parallel...

---

## Phase 2: Launch 10 Specialist Agents

Launch ALL 10 agents in a **single message with 10 parallel Task tool calls**. Do NOT use `run_in_background` — launch as regular parallel calls so all results arrive together before synthesis.

If the system limits parallel Task calls, launch in two batches of 5. If any agent fails or returns an error, note the failure in the coordinator prompt and proceed with the remaining agents' reports.

The log format is: `TIMESTAMP [LEVEL] [CORRELATION_ID] MODULE: MESSAGE`
Example: `2026-02-20 21:11:44 [INFO] [-] src.services.llm_client: LLM call complete: model=phi4-mini:latest, schema=ItemQualityScores, 9.89s, tokens: 668+202=870`

**Shared agent rules (apply to ALL 10 agents):**

1. **Grep syntax**: The Grep tool uses **ripgrep**. Use `|` for alternation (NOT `\|` which matches a literal pipe in ripgrep). For case-insensitive searches, use the `-i` flag (e.g., `pattern: "failed", -i: true`) instead of listing all case variants manually.
2. **Large result sets**: If a grep pattern returns more than 100 matches, sample the first 20, last 20, and 10 random matches from the middle. Report the total count and note that sampling was used.
3. **Output constraint**: Keep your report under **2,000 words**. Group similar findings rather than listing each individually. **Do not include raw log snippets** — use line-number references (e.g., "Lines 1042-1058") so the coordinator can request details if needed. The coordinator must fit all 10 reports in its context window.
4. **Severity vs action**: Severity is `CRITICAL/HIGH/MEDIUM/LOW`. Action type is `BUGFIX/INVESTIGATE/CONFIGURE/REFACTOR`. These are independent — report both per finding.

---

### Agent 1: Error & Warning Triage

| Setting | Value |
|---------|-------|
| name | error-warning-triage |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are analyzing the Story Factory application log for errors, warnings, and silent failures.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Find all errors and warnings:**
- Grep for `\[ERROR\]` with `-C: 15` (15 lines of surrounding context per match)
- Grep for `\[WARNING\]` with `-C: 5` (5 lines of surrounding context per match)

**Step 2 — Analyze each finding:**
For each ERROR:
- Is the error handled or does it propagate silently?
- Is the same error logged multiple times (duplicate logging)?
- How many log lines does the traceback consume (log bloat)?
- What user-visible impact does this have?

For each WARNING:
- Is this a one-off or a recurring pattern? Count occurrences.
- Is the warning actionable or just noise?
- Does anything go wrong AFTER the warning (check next 20 lines)?
- Are warnings being silently swallowed without user notification?

**Step 3 — Hunt for silent failures:**
- Grep for `failed|Failed|FAILED` at INFO/DEBUG level — these are failures not promoted to ERROR
- Grep for `skip|Skip|SKIP` — things being silently skipped
- Grep for `= None|=None|returned None|got None|value.*empty|result.*null` at WARNING/ERROR level — targeted patterns where a missing value indicates a real problem
- Grep for `retry|Retry|fallback|Fallback` — recovery paths that might mask real issues

**Output format:**
Return your findings as a markdown report with this structure:

```markdown
## Error & Warning Triage Findings

### Summary
- Total errors: N
- Total warnings: N
- Silent failures detected: N
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings

#### [SEVERITY] Title
- **Evidence:** Lines X-Y (description of what those lines show)
- **Impact:** What this breaks or risks
- **Root cause:** Module path and likely source file
- **Action type:** BUGFIX / INVESTIGATE / CONFIGURE / REFACTOR
- **Recommendation:** Specific fix or next steps
```

---

### Agent 2: Performance & Timing

| Setting | Value |
|---------|-------|
| name | performance-timing |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are analyzing the Story Factory application log for performance issues, slow operations, and timing anomalies.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Extract all LLM call timings:**
- Grep for `LLM call complete` — extract model, schema, duration, token counts
- Grep for `cold-start|cold start|not loaded` — model loading penalties
- Grep for `Timeout|timeout|timed out` — timeout events
- Grep for `Backing off|Retrying in` — retry delays

**Step 2 — Extract operation timings:**
- Grep for lines containing timing patterns like `\d+\.\d+s` or `took \d+`
- Grep for `complete|completed|finished|done` with timing info
- Grep for `Starting|starting` and correlate with completion times

**Step 3 — Analyze:**
- Calculate per-model statistics: avg/min/max duration, total token usage
- Identify outlier calls (>2x the average for that model)
- Check for timeout proximity (calls that took >80% of their timeout)
- Identify cold-start penalties and their impact on total time
- Calculate total LLM wall-clock time vs session duration (utilization)
- Find operations that took disproportionately long for their output

**Step 4 — Check for bottlenecks:**
- Are there sequential LLM calls that could be parallelized?
- Long gaps between operations (idle time)?
- Operations blocking on model loading?

**Output format:**
Return your findings as a markdown report. Include a statistics table:

```markdown
## Performance & Timing Findings

### LLM Call Statistics
| Model | Calls | Avg (s) | Min (s) | Max (s) | Total Tokens | Avg Tokens/Call |
|-------|-------|---------|---------|---------|-------------|----------------|

### Summary
- Total LLM wall-clock time: Xs
- Session duration: Xs
- LLM utilization: X%
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings

#### [SEVERITY] Title
- **Evidence:** ...
- **Impact:** ...
- **Root cause:** ...
- **Action type:** BUGFIX / INVESTIGATE / CONFIGURE / REFACTOR
- **Recommendation:** ...
```

---

### Agent 3: Quality Loop Patterns

| Setting | Value |
|---------|-------|
| name | quality-loop-patterns |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are analyzing the Story Factory application log for quality refinement loop patterns, scoring issues, and wasted refinement work.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Extract all quality loop data:**
- Grep for `quality refinement loop|Starting quality refinement` — loop starts
- Grep for `iteration.*score|scoring round` — per-iteration scores
- Grep for `dimension scores` — detailed dimension breakdowns
- Grep for `plateaued|plateau|Early stop` — early stopping events
- Grep for `hail-mary|hail.mary` — fresh creation attempts
- Grep for `threshold not met|did not meet quality threshold` — threshold failures
- Grep for `REFINEMENT ANALYTICS` — read the full multiline analytics blocks (read 15 lines after each match)
- Grep for `Batch.*summary` — batch-level summaries

**Step 2 — Reconstruct each entity's refinement journey:**
For each entity that went through refinement:
- Score progression across iterations (e.g., 6.4 -> 7.0 -> 6.6)
- Did it improve, plateau, or regress?
- Was a hail-mary attempted? Did it help?
- Final outcome: threshold met or best-effort returned?
- Total time spent on this entity

**Step 3 — Analyze patterns:**
- Which entity types have the worst pass rates?
- Are thresholds appropriately set? (If >50% of entities fail threshold, it might be too high)
- How often do hail-marys beat the original? (If rarely, it's wasted compute)
- Score regressions: how often does refinement make things WORSE?
- Dimension analysis: which dimensions consistently score low?
- Is there a pattern of diminishing returns (iteration 2 always close to iteration 1)?

**Output format:**
Return your findings as a markdown report. Include a quality loop summary table:

```markdown
## Quality Loop Pattern Findings

### Quality Loop Summary
| Entity Type | Count | Pass Rate | Avg Iters | Avg Score | Threshold | Hail-Marys | Hail-Mary Win Rate |
|------------|-------|-----------|-----------|-----------|-----------|------------|-------------------|

### Summary
- Total refinement cycles: N
- Overall pass rate: X%
- Total refinement time: Xs
- Wasted time on failed entities: Xs
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings
...
```

---

### Agent 4: Data Integrity & Parsing

| Setting | Value |
|---------|-------|
| name | data-integrity |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are analyzing the Story Factory application log for data corruption, parsing errors, and integrity issues.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Timestamp/temporal parsing:**
- Grep for `Parsing timestamp` — extract the input and result for each parse
- Grep for `Extracted year` — check if years match the input (especially negative years!)
- Grep for `Parsed timestamp result` — verify year/month/day fields are correct
- CRITICAL: Check for negative year handling. Input `{"year": -10}` should produce year=-10, NOT year=None or year=10. Check EVERY parsed timestamp for sign correctness.
- Check for era_name fields: are they being populated when the input contains era info?

**Step 2 — Entity data quality:**
- Grep for `nesting depth|flattening|Attributes exceed` — attribute structure violations
- Grep for `add_entity|Added entity` — entity creation events, check for missing fields
- Grep for `lifecycle` — lifecycle extraction, check for entities with no lifecycle data
- Grep for `embedding.*empty|embedding.*failed|embedding.*None` — embedding failures

**Step 3 — Validation and uniqueness:**
- Grep for `Validating name|is unique|duplicate|already exists` — name collision handling
- Grep for `semantic match.*above threshold` — near-duplicate entities that got through
- Grep for `similarity` — check similarity scores for suspiciously high values

**Step 4 — Cross-reference:**
- For entities where temporal data was lost/corrupted, check if quality scores were affected
- For entities with flattened attributes, check if important data was lost in flattening

**Output format:**
Return findings as markdown. Pay special attention to data corruption where the log shows "success" but the data is actually wrong (e.g., year=-10 parsed as year=None looks like a successful parse but is data loss).

```markdown
## Data Integrity Findings

### Summary
- Timestamps parsed: N
- Parsing errors/data loss: N
- Attribute violations: N
- Embedding failures: N
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings
...
```

---

### Agent 5: Redundancy & Waste

| Setting | Value |
|---------|-------|
| name | redundancy-waste |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are analyzing the Story Factory application log for redundant operations, wasted work, and inefficiencies.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Find repeated operations:**
- Grep for `get_vram called` — count how many times VRAM is checked. Is the value ever different?
- Grep for `check_health called` — count health checks. Are they redundant within short time windows?
- Grep for `list_installed|Found.*installed models` — how many times is the model list fetched?
- Grep for `get_model_info called` — count per-model info calls. Could these be batched?
- Grep for `Refreshed environment context` — count cache refreshes. Do values change?
- Grep for `Loaded per-entity quality thresholds` — is this loaded repeatedly vs once?
- Grep for `Building JudgeConsistencyConfig` — is this rebuilt every call?

**Step 2 — Quantify waste:**
For each repeated operation:
- How many times does it repeat?
- Over what time window?
- Does the result ever change between repetitions?
- If the result never changes, calculate wasted time (count * avg_duration)

**Step 3 — Cache effectiveness:**
- Grep for `cache hit|cache miss|Cache hit|Cache miss` — track hit/miss ratio
- Grep for `cached|Using cached` — what's being cached effectively?
- Grep for `Refreshed|refresh` — are caches being invalidated unnecessarily?

**Step 4 — Log verbosity waste:**
- Identify log messages that repeat identically more than 5 times
- Calculate how many log lines are consumed by redundant messages
- Check if DEBUG-level messages are providing value or just noise

**Output format:**

```markdown
## Redundancy & Waste Findings

### Summary
- Redundant operations detected: N
- Estimated wasted time: Xs
- Cache hit rate: X%
- Redundant log lines: N (X% of total)
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings
...
```

---

### Agent 6: Sequence & State Transitions

| Setting | Value |
|---------|-------|
| name | sequence-state |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are analyzing the Story Factory application log for operation ordering issues, state machine violations, and unexpected transitions.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Reconstruct the operation sequence:**
- Grep for `Initializing|initialized|Starting|starting|Complete|complete` — lifecycle events
- Grep for `Rendering.*page|Rendering.*at` — page navigation sequence
- Grep for `Log level changed` — settings changes mid-session
- Grep for `clicked|confirmed|User` — user actions

**Step 2 — Check operation ordering:**
- Does initialization happen in the expected order? (settings -> services -> templates -> UI)
- Are there operations that happen before their prerequisites?
- Do correlation IDs switch unexpectedly mid-operation?
- Are there "orphaned" start events without corresponding completion events?

**Step 3 — World build pipeline ordering:**
- Grep for the build pipeline stages: calendar, characters, chapters, locations, factions, items, concepts, relationships, events, embeddings
- Verify they happen in the documented order
- Check for stages that start before the previous stage completed
- Look for stages that are skipped or repeated

**Step 4 — State consistency:**
- Grep for `project.*list|Listed.*projects|Refreshed.*cache` — project state changes
- Check if settings are re-loaded during operations (unexpected mid-operation config changes)
- Look for operations on entities that don't exist yet or have been deleted

**Output format:**

```markdown
## Sequence & State Findings

### Session Timeline
| Time | Event | Notes |
|------|-------|-------|

### Summary
- State violations detected: N
- Ordering issues: N
- Orphaned operations: N
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings
...
```

---

### Agent 7: General Anomaly Detector

| Setting | Value |
|---------|-------|
| name | anomaly-detector |
| model | opus |
| subagent_type | Explore |

**Prompt:**

You are the anomaly detector for the Story Factory application log. Your job is to find things that the other 9 specialist agents might miss — the "unknown unknowns." Things that look correct on the surface but are actually wrong. Patterns that are unusual but not obviously broken. Cross-domain correlations that only appear when you read the log holistically.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Your approach — read broadly, think deeply:**

1. Read the first 100 lines to understand the session context (app startup, configuration)
2. Read 5-6 evenly spaced 100-line samples from different parts of the log. Calculate offsets by dividing `<LINE_COUNT>` into 6 equal segments and reading 100 lines starting at each segment boundary (e.g., for a 6000-line log: offsets 0, 1000, 2000, 3000, 4000, 5000)
3. Read the last 100 lines to see the session end state

**For each section you read, ask yourself:**
- Does anything seem "off" even though it logged at INFO/DEBUG level?
- Are there numbers that don't add up?
- Are there operations that succeed but with suspicious characteristics?
- Are there patterns that SHOULD be in the log but are MISSING?
- Does the log tell a coherent story, or are there narrative gaps?

**Specific things to hunt for:**
- Entity names or data that look like LLM hallucinations (nonsensical names, contradictory attributes)
- Quality scores that seem too high or too low for the apparent content quality
- Operations that complete "successfully" but produce empty or minimal output
- Time gaps that are unexplained (no log entries for extended periods)
- Correlation ID mismatches (operations attributed to wrong sessions)
- Configuration values that seem wrong for the context
- Log messages that contradict each other
- Patterns that suggest the LLM is gaming the quality scoring system
- Interleaved log messages from different threads suggesting race conditions
- Lock contention or deadlock patterns (operations stalling without explanation)
- Operations on the same entity from different correlation IDs simultaneously
- Anything that makes you think "wait, that can't be right"

**You have no predetermined grep patterns.** You are the fresh eyes. Read the log and report what jumps out at you.

**Output format:**

```markdown
## Anomaly Detector Findings

### Summary
- Sections sampled: N (of M total lines)
- Anomalies detected: N
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings

#### [SEVERITY] Title
- **Evidence:** Lines X-Y (description of what those lines show)
- **Impact:** What this breaks or risks
- **Root cause:** Best guess at the source, or "unknown — needs investigation"
- **Why this is anomalous:** Explain what you expected vs what you found
- **Possible explanations:** What could cause this (ranked by likelihood)
- **Action type:** BUGFIX / INVESTIGATE / CONFIGURE / REFACTOR
- **Recommendation:** Specific next steps
```

---

### Agent 8: LLM Token Economics

| Setting | Value |
|---------|-------|
| name | token-economics |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are analyzing the Story Factory application log for LLM token usage patterns, prompt efficiency, and token waste.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Extract all token data:**
- Grep for `tokens:` — extract prompt+completion=total for every LLM call
- Grep for `LLM call complete` — correlate tokens with model, schema, duration
- Grep for `Stream consumed.*chunks.*chars` — output size data

**Step 2 — Calculate token economics:**
Per model:
- Total input tokens, total output tokens, total tokens
- Average input/output ratio (high ratio = bloated prompts relative to output)
- Tokens per second (throughput)
- Cost per entity type (if identifiable from schema names)

Per entity type (schema):
- Average tokens per generation (e.g., Item vs Faction vs ChapterQualityScores)
- Which entity types are most expensive?
- Are judge calls appropriately cheaper than creator calls?

**Step 3 — Find waste:**
- Token usage on failed attempts (calls that were retried or whose output was discarded)
- Hail-mary calls that didn't improve scores (tokens with zero value)
- Refinement iterations where score didn't improve (wasted refinement tokens)
- Disproportionate prompt sizes (if input >> output consistently, prompts may be bloated)

**Step 4 — Efficiency patterns:**
- Are smaller models used for simpler tasks? (judge with phi4-mini vs creator with mistral-small)
- Token-to-quality ratio: do more tokens produce better scores?
- Compare structured output calls vs free-text calls in token efficiency

**Output format:**

```markdown
## Token Economics Findings

### Token Usage by Model
| Model | Calls | Input Tokens | Output Tokens | Total | Avg Input/Output Ratio | Tokens/sec |
|-------|-------|-------------|--------------|-------|----------------------|-----------|

### Token Usage by Schema
| Schema | Calls | Avg Tokens | Avg Duration |
|--------|-------|-----------|-------------|

### Summary
- Total tokens consumed: N
- Estimated wasted tokens: N (X%)
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings
...
```

---

### Agent 9: Model Selection & Switching

| Setting | Value |
|---------|-------|
| name | model-selection |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are analyzing the Story Factory application log for model selection patterns, model switching costs, and model-role assignment effectiveness.

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Map model-role assignments:**
- Grep for `creator model|Creator model|Using cached creator` — which models serve creator role
- Grep for `judge model|Judge model|Using cached judge` — which models serve judge role
- Grep for `architect|Architect` in model context — architect model
- Grep for `model=` — extract all model names used across all calls

**Step 2 — Analyze model switching:**
- Track the sequence of model= values across LLM calls
- How often does the active model switch? (Each switch = potential cold-start)
- Grep for `not loaded|cold-start|loading model` — explicit cold-start events
- Calculate time between model switches (if <60s, might overlap with unloading)

**Step 3 — Model-quality correlation:**
- For each model, what are the average quality scores it produces?
- Do certain models consistently score below threshold for certain entity types?
- Compare the judge model's scoring patterns: is it consistent, or does it vary wildly?
- Are there patterns where switching to a different model mid-pipeline affects quality?

**Step 4 — Model sizing:**
- Grep for `size=|VRAM=|vram=` — model sizes and VRAM availability
- Is the GPU being underutilized (small model on large GPU)?
- Are any models close to the 80% GPU residency threshold?
- Grep for `Timeout for` — are timeouts correctly scaled to model size?

**Output format:**

```markdown
## Model Selection Findings

### Model-Role Map
| Model | Size | Roles | Calls | Avg Score (as creator) | Avg Score (as judge) |
|-------|------|-------|-------|----------------------|---------------------|

### Summary
- Models used: N
- Model switches: N
- Cold-start events: N
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings
...
```

---

### Agent 10: User Session Reconstruction

| Setting | Value |
|---------|-------|
| name | session-reconstruction |
| model | sonnet |
| subagent_type | Explore |

**Prompt:**

You are reconstructing the user's session from the Story Factory application log. Your goal is to understand the human experience — what did the user do, how long did they wait, and where was the experience bad?

Log file: `<LOG_FILE>` (<LINE_COUNT> lines)

**Step 1 — Build the session timeline:**
- Read the first 50 lines (app startup)
- Grep for `Rendering.*page|page at` — page navigation events
- Grep for `clicked|confirmed|User|user` — user actions
- Grep for `Log level changed|Settings saved` — settings changes
- Grep for `Rebuild|rebuild|Build|build.*world|clear.*world|Clear.*World` — major user-initiated operations
- Read the last 50 lines (session end)

**Step 2 — Identify wait times:**
For each user action, calculate:
- How long until the next visible response (next page render, next UI event)?
- How long did background operations take (world build, story generation)?
- Were there long gaps where the user was waiting with no feedback?

**Step 3 — Identify UX pain points:**
- Operations that took >30 seconds with no intermediate progress logging
- User actions that triggered unexpectedly expensive backend work
- Settings changes that didn't take effect (changed then reverted?)
- Pages that took multiple seconds to render
- Operations the user started but didn't complete (abandoned?)

**Step 4 — Session narrative:**
Write a short narrative of what happened in this session:
- What was the user trying to do?
- How long did it take?
- What went well?
- What was frustrating?

**Output format:**

The Session Timeline and Session Narrative sections below are supplementary context. Your primary deliverable is the standard Findings section with severity-tagged findings.

```markdown
## Session Reconstruction Findings

### Session Timeline
| Time | Event | Wait Time | Notes |
|------|-------|-----------|-------|

### Session Narrative
[2-3 paragraphs describing what happened]

### Summary
- Session duration: Xs
- Total user wait time: Xs
- Longest single wait: Xs (for what operation?)
- Pages visited: N
- Major operations: N
- Findings: N (critical: N, high: N, medium: N, low: N)
- Items needing investigation: N

### Findings

#### [SEVERITY] Title
- **Evidence:** Lines X-Y (description of what those lines show)
- **Impact:** What this breaks or risks for the user
- **Root cause:** Module path and likely source file
- **Action type:** BUGFIX / INVESTIGATE / CONFIGURE / REFACTOR
- **Recommendation:** Specific fix or next steps
```

---

## Phase 3: Coordinator Synthesis

After ALL 10 agents have returned their findings, launch a single **Opus** coordinator agent using the Task tool (subagent_type: `Explore`, model: `opus`). The coordinator only needs read access to synthesize findings — it does not write files.

The coordinator should return the full report as its **output text**. Do NOT instruct the coordinator to write files — file writing happens in Phase 4.

**Coordinator prompt:**

You are the coordinator for a 10-agent log analysis. You have received reports from these specialists:

1. Error & Warning Triage
2. Performance & Timing
3. Quality Loop Patterns
4. Data Integrity & Parsing
5. Redundancy & Waste
6. Sequence & State Transitions
7. General Anomaly Detector
8. LLM Token Economics
9. Model Selection & Switching
10. User Session Reconstruction

**Your tasks:**

**1. Deduplicate:** Multiple agents may have flagged the same issue. Merge duplicates, keeping the most detailed evidence and noting all source agents.

**2. Cross-reference:** Connect findings across domains. Examples:
- Data integrity issues (bad timestamp parsing) may explain quality loop failures (low scores on those entities)
- Redundant operations (excessive VRAM checks) may explain performance issues (slow page loads)
- Model selection issues may explain token waste patterns

**3. Prioritize:** Assign final severity based on combined evidence:
- **CRITICAL** — Data corruption, silent data loss, security issues
- **HIGH** — Significant wasted compute, user-facing performance issues, systematic failures
- **MEDIUM** — Redundancy, suboptimal patterns, moderate inefficiency
- **LOW** — Log noise, minor optimization opportunities, cosmetic issues

**4. Categorize actions:**
- `BUGFIX` — Clear code bug with identifiable root cause, can be fixed directly
- `INVESTIGATE` — Suspicious pattern needing deeper analysis or investigation script
- `CONFIGURE` — Settings/threshold tuning, not a code change
- `REFACTOR` — Code improvement for efficiency/maintainability

**5. Return the final report** as your output in this format:

```markdown
# Log Analysis Report

**Log file:** `<LOG_FILE>`
**Lines analyzed:** <LINE_COUNT>
**Analysis date:** <DATE>
**Agents used:** 10 specialists (9 Sonnet, 1 Opus) + 1 Opus coordinator

## Executive Summary

- **Critical:** N findings
- **High:** N findings
- **Medium:** N findings
- **Low:** N findings
- **Investigate:** N items needing human judgment

**Top issues:**
1. [One-line summary of most important finding]
2. [One-line summary of second most important finding]
3. [One-line summary of third most important finding]

**Quick wins** (easy fixes with high impact):
1. ...
2. ...
3. ...

## Findings by Severity

### CRITICAL

#### C1: [Title] [ACTION_TYPE]
- **Source agent(s):** [which agents found this]
- **Evidence:** Lines X-Y (description of what those lines show)
- **Impact:** [what this breaks or risks]
- **Root cause:** `<source_file_path>` — [description]
- **Recommendation:** [specific fix or investigation steps]
- **Cross-references:** [related findings from other agents, if any]

[Continue for all CRITICAL findings...]

### HIGH
[Same format, numbered H1, H2, ...]

### MEDIUM
[Same format, numbered M1, M2, ...]

### LOW
[Same format, numbered L1, L2, ...]

## Items to Investigate

### I1: [Title]
- **Source agent(s):** [which agents flagged this]
- **Why this needs investigation:** [what's suspicious but not proven]
- **Suggested approach:** [how to investigate — e.g., "create investigation script", "check database", "reproduce with specific input"]

[Continue for all INVESTIGATE items...]

## Statistics

### LLM Call Performance
[Include the performance agent's statistics table]

### Quality Loop Efficiency
[Include the quality loop agent's summary table]

### Token Economics
[Include the token economics agent's usage tables]

### Session Timeline
[Include the session reconstruction agent's timeline]

## Agent Coverage Report

| Agent | Lines Analyzed | Findings | Critical | High | Medium | Low | Investigate |
|-------|---------------|----------|----------|------|--------|-----|------------|
| Error & Warning Triage | ... | ... | ... | ... | ... | ... | ... |
[Complete for all 10 agents]

| **Totals (before dedup)** | | ... | ... | ... | ... | ... | ... |
| **Final (after dedup)** | | ... | ... | ... | ... | ... | ... |
```

Pass ALL 10 agent reports to the coordinator in its prompt, clearly labeled by agent name. Agent reports use line-number references instead of raw log snippets — the coordinator should preserve these references in its output. If any agent failed, note the failure in the coordinator prompt so the report reflects incomplete coverage.

---

## Phase 4: Present Results

After receiving the coordinator's output:

1. **Write the report** to `<OUTPUT_PATH>` using the Write tool
2. Print the **Executive Summary** section to the user inline (critical/high/medium/low counts + top 3 issues + quick wins)
3. Tell the user: "Full report saved to `<OUTPUT_PATH>`"
4. List all CRITICAL and HIGH findings as a numbered list with one-line summaries
5. Ask the user if they want to start fixing any of the BUGFIX items directly
