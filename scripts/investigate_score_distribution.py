#!/usr/bin/env python3
"""Score Distribution Analysis — parses quality loop logs to diagnose judge bias.

Reads output/logs/story_factory.log (or a user-specified log file) and extracts
quality-loop scoring data to identify:

1. Score clustering — are scores bunched in a narrow range?
2. Per-dimension variance — does the judge differentiate between dimensions?
3. Generation time vs score correlation — do fast entities score as high as slow ones?
4. Per-entity-type breakdowns — which entity types cluster most?

Log patterns parsed (from _quality_loop.py):
  - "<Type> creation took <N>s (iteration <M>)"
  - "<Type> '<name>' iteration <N>: score <X> (best so far: <Y> at iteration <Z>)"
  - "<Type> '<name>' iteration <N> dimension scores: {dim: score, ...}"
  - "<Type> '<name>' met quality threshold (<X> >= <Y>)"
  - "<Type> refinement took <N>s (iteration <M>)"

Usage:
    python scripts/investigate_score_distribution.py [options]
      --log-file path/to/log   (default: output/logs/story_factory.log)
      --output results.json    (default: printed summary only)
      --verbose                (print per-entity details)
"""

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =====================================================================
# Log line patterns
# =====================================================================

# "Faction creation took 2.34s (iteration 1)"
CREATION_TIME_RE = re.compile(
    r"(\w+) creation took ([\d.]+)s \(iteration (\d+)\)",
)

# "Faction refinement took 3.12s (iteration 2)"
REFINEMENT_TIME_RE = re.compile(
    r"(\w+) refinement took ([\d.]+)s \(iteration (\d+)\)",
)

# "Faction 'The Veilkeepers' iteration 1: score 7.5 (best so far: 7.5 at iteration 1)"
ITERATION_SCORE_RE = re.compile(
    r"(\w+) '(.+?)' iteration (\d+): score ([\d.]+) "
    r"\(best so far: ([\d.]+) at iteration (\d+)\)",
)

# "Faction 'The Veilkeepers' iteration 1 dimension scores: {'coherence': '8.5', ...}"
DIMENSION_SCORES_RE = re.compile(
    r"(\w+) '(.+?)' iteration (\d+) dimension scores: (\{.+\})",
)

# "Faction 'The Veilkeepers' met quality threshold (7.5 >= 7.5)"
THRESHOLD_MET_RE = re.compile(
    r"(\w+) '(.+?)' met quality threshold \(([\d.]+) >= ([\d.]+)\)",
)


# =====================================================================
# Data structures
# =====================================================================
class EntityRecord:
    """Collects all scoring data for a single entity across iterations."""

    def __init__(self, entity_type: str, name: str) -> None:
        self.entity_type = entity_type.lower()
        self.name = name
        self.scores: list[float] = []
        self.dimension_scores: list[dict[str, float]] = []
        self.creation_time: float | None = None
        self.refinement_times: list[float] = []
        self.threshold_met: bool = False
        self.threshold_value: float | None = None
        self.final_score: float | None = None

    @property
    def best_score(self) -> float | None:
        """Return the highest average score across iterations."""
        return max(self.scores) if self.scores else None

    @property
    def total_generation_time(self) -> float:
        """Total time spent creating + refining this entity."""
        creation = self.creation_time or 0.0
        refinements = sum(self.refinement_times)
        return creation + refinements


# =====================================================================
# Parsing
# =====================================================================
def parse_log_file(log_path: Path) -> list[EntityRecord]:
    """Parse a story_factory.log file and extract entity scoring records.

    Args:
        log_path: Path to the log file.

    Returns:
        List of EntityRecord objects with scoring data populated.
    """
    logger.info("Parsing log file: %s", log_path)

    # Key entities by (type, name) for aggregation
    entities: dict[tuple[str, str], EntityRecord] = {}

    def get_or_create(entity_type: str, name: str) -> EntityRecord:
        key = (entity_type.lower(), name)
        if key not in entities:
            entities[key] = EntityRecord(entity_type, name)
        return entities[key]

    line_count = 0
    matched_lines = 0

    # Pending creation/refinement times keyed by entity type (lowercase).
    # Creation and refinement log lines appear *before* the iteration score
    # line that contains the entity name, so we stash the duration here and
    # resolve it when the score line arrives.
    pending_creation: dict[str, float] = {}
    pending_refinement: dict[str, float] = {}

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1

            # Creation time
            m = CREATION_TIME_RE.search(line)
            if m:
                entity_type, duration = m.group(1), float(m.group(2))
                pending_creation[entity_type.lower()] = duration
                matched_lines += 1
                continue

            # Refinement time
            m = REFINEMENT_TIME_RE.search(line)
            if m:
                entity_type, duration = m.group(1), float(m.group(2))
                pending_refinement[entity_type.lower()] = duration
                matched_lines += 1
                continue

            # Iteration score (most important line)
            m = ITERATION_SCORE_RE.search(line)
            if m:
                entity_type = m.group(1)
                name = m.group(2)
                score = float(m.group(4))

                record = get_or_create(entity_type, name)
                record.scores.append(score)

                # Resolve pending creation time
                if entity_type.lower() in pending_creation:
                    if record.creation_time is None:
                        record.creation_time = pending_creation.pop(entity_type.lower())
                    else:
                        pending_creation.pop(entity_type.lower())

                # Resolve pending refinement time
                if entity_type.lower() in pending_refinement:
                    record.refinement_times.append(pending_refinement.pop(entity_type.lower()))

                matched_lines += 1
                continue

            # Dimension scores
            m = DIMENSION_SCORES_RE.search(line)
            if m:
                entity_type = m.group(1)
                name = m.group(2)
                scores_str = m.group(4)

                record = get_or_create(entity_type, name)
                try:
                    # The log format uses {'key': 'val'} — single quotes, need eval-safe parse
                    dim_scores = _parse_dimension_dict(scores_str)
                    record.dimension_scores.append(dim_scores)
                except (ValueError, SyntaxError) as e:
                    logger.warning("Failed to parse dimension scores: %s", e)

                matched_lines += 1
                continue

            # Threshold met
            m = THRESHOLD_MET_RE.search(line)
            if m:
                entity_type = m.group(1)
                name = m.group(2)
                final_score = float(m.group(3))
                threshold = float(m.group(4))

                record = get_or_create(entity_type, name)
                record.threshold_met = True
                record.final_score = final_score
                record.threshold_value = threshold
                matched_lines += 1
                continue

    logger.info(
        "Parsed %d lines, matched %d scoring-related lines, found %d entities",
        line_count,
        matched_lines,
        len(entities),
    )
    return list(entities.values())


def _parse_dimension_dict(s: str) -> dict[str, float]:
    """Parse a dimension scores dict string from log output.

    The log format uses Python dict repr with single-quoted string values:
    {'coherence': '8.5', 'influence': '7.0', ...}

    Args:
        s: String representation of the dimension scores dict.

    Returns:
        Dict mapping dimension names to float scores.
    """
    # Replace single quotes with double quotes for JSON parsing
    cleaned = s.replace("'", '"')
    raw = json.loads(cleaned)
    return {k: float(v) for k, v in raw.items()}


# =====================================================================
# Analysis functions
# =====================================================================
def compute_score_histogram(records: list[EntityRecord]) -> dict[str, int]:
    """Compute histogram of all average scores across all entities.

    Bins scores into 0.5-wide buckets from 0.0 to 10.0.

    Args:
        records: Entity records with scores.

    Returns:
        Dict mapping bin labels (e.g. "7.0-7.5") to count of scores in that bin.
    """
    bins: dict[str, int] = {}
    for r in records:
        for score in r.scores:
            bin_lower = math.floor(score * 2) / 2  # 0.5-wide bins
            bin_label = f"{bin_lower:.1f}-{bin_lower + 0.5:.1f}"
            bins[bin_label] = bins.get(bin_label, 0) + 1

    # Sort by bin lower bound
    return dict(sorted(bins.items(), key=lambda x: float(x[0].split("-")[0])))


def compute_clustering_metrics(records: list[EntityRecord]) -> dict[str, Any]:
    """Compute score clustering metrics.

    Args:
        records: Entity records with scores.

    Returns:
        Dict with:
        - unique_scores: number of distinct score values
        - total_scores: total number of scores
        - modal_scores: top 3 most common scores
        - score_range: max - min of all scores
        - std_dev: standard deviation of all scores
        - iqr: interquartile range
    """
    all_scores: list[float] = []
    for r in records:
        all_scores.extend(r.scores)

    if not all_scores:
        return {
            "unique_scores": 0,
            "total_scores": 0,
            "modal_scores": [],
            "score_range": 0.0,
            "std_dev": 0.0,
            "iqr": 0.0,
        }

    # Round to 1 decimal for unique counting (matches log format)
    rounded = [round(s, 1) for s in all_scores]
    score_counts: dict[float, int] = defaultdict(int)
    for s in rounded:
        score_counts[s] += 1

    sorted_counts = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)
    modal_scores = [{"score": s, "count": c} for s, c in sorted_counts[:5]]

    mean = sum(all_scores) / len(all_scores)
    variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
    std_dev = math.sqrt(variance)

    sorted_scores = sorted(all_scores)
    q1_idx = len(sorted_scores) // 4
    q3_idx = 3 * len(sorted_scores) // 4
    iqr = sorted_scores[q3_idx] - sorted_scores[q1_idx] if len(sorted_scores) >= 4 else 0.0

    return {
        "unique_scores": len(score_counts),
        "total_scores": len(all_scores),
        "modal_scores": modal_scores,
        "score_range": round(max(all_scores) - min(all_scores), 2),
        "std_dev": round(std_dev, 3),
        "iqr": round(iqr, 2),
        "mean": round(mean, 2),
        "min": round(min(all_scores), 1),
        "max": round(max(all_scores), 1),
    }


def compute_per_type_breakdown(
    records: list[EntityRecord],
) -> dict[str, dict[str, Any]]:
    """Compute per-entity-type score statistics.

    Args:
        records: Entity records with scores.

    Returns:
        Dict mapping entity type to stats dict with mean, std_dev, count,
        avg_creation_time, avg_total_time, threshold_met_rate.
    """
    type_groups: dict[str, list[EntityRecord]] = defaultdict(list)
    for r in records:
        type_groups[r.entity_type].append(r)

    result: dict[str, dict[str, Any]] = {}
    for entity_type, group in sorted(type_groups.items()):
        all_scores: list[float] = []
        for r in group:
            if r.scores:
                all_scores.append(r.scores[-1])  # Final score

        creation_times = [r.creation_time for r in group if r.creation_time is not None]
        total_times = [r.total_generation_time for r in group if r.total_generation_time > 0]
        met_count = sum(1 for r in group if r.threshold_met)

        mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
        variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores) if all_scores else 0.0

        result[entity_type] = {
            "entity_count": len(group),
            "score_count": len(all_scores),
            "mean_final_score": round(mean, 2),
            "std_dev": round(math.sqrt(variance), 3),
            "min_score": round(min(all_scores), 1) if all_scores else None,
            "max_score": round(max(all_scores), 1) if all_scores else None,
            "avg_creation_time": (
                round(sum(creation_times) / len(creation_times), 2) if creation_times else None
            ),
            "avg_total_time": (
                round(sum(total_times) / len(total_times), 2) if total_times else None
            ),
            "threshold_met_rate": (round(met_count / len(group), 2) if group else 0.0),
        }

    return result


def compute_dimension_variance(records: list[EntityRecord]) -> dict[str, dict[str, Any]]:
    """Compute per-dimension score variance across all entities.

    Answers: does the judge give different scores to different dimensions,
    or are all dimensions scored identically?

    Args:
        records: Entity records with dimension_scores populated.

    Returns:
        Dict mapping entity type to dimension stats (mean, std_dev per dimension,
        plus intra_entity_spread — the average range within a single entity's dimensions).
    """
    type_dims: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    intra_spreads: dict[str, list[float]] = defaultdict(list)

    for r in records:
        for dim_scores in r.dimension_scores:
            for dim, score in dim_scores.items():
                type_dims[r.entity_type][dim].append(score)

            # Intra-entity spread: how much do dimensions differ within one judge call?
            dim_values = list(dim_scores.values())
            if len(dim_values) >= 2:
                spread = max(dim_values) - min(dim_values)
                intra_spreads[r.entity_type].append(spread)

    result: dict[str, dict[str, Any]] = {}
    for entity_type, dims in sorted(type_dims.items()):
        dim_stats: dict[str, dict[str, float]] = {}
        for dim, values in sorted(dims.items()):
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
            dim_stats[dim] = {
                "mean": round(mean, 2),
                "std_dev": round(math.sqrt(variance), 3),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
                "count": len(values),
            }

        spreads = intra_spreads.get(entity_type, [])
        avg_intra_spread = round(sum(spreads) / len(spreads), 2) if spreads else 0.0

        result[entity_type] = {
            "dimensions": dim_stats,
            "avg_intra_entity_spread": avg_intra_spread,
        }

    return result


def compute_time_vs_score(records: list[EntityRecord]) -> dict[str, Any]:
    """Compute correlation between generation time and final score.

    Args:
        records: Entity records with timing and score data.

    Returns:
        Dict with per-type time-vs-score data and overall Pearson correlation.
    """
    pairs: list[tuple[float, float]] = []
    type_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for r in records:
        if r.total_generation_time > 0 and r.scores:
            final_score = r.scores[-1]
            pairs.append((r.total_generation_time, final_score))
            type_pairs[r.entity_type].append((r.total_generation_time, final_score))

    def _pearson(data: list[tuple[float, float]]) -> float:
        if len(data) < 3:
            return 0.0
        n = len(data)
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in data) / n
        std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
        std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)
        if std_x == 0 or std_y == 0:
            return 0.0
        return round(cov / (std_x * std_y), 3)

    per_type: dict[str, dict[str, Any]] = {}
    for entity_type, tp in sorted(type_pairs.items()):
        times = [t for t, _ in tp]
        scores = [s for _, s in tp]
        per_type[entity_type] = {
            "count": len(tp),
            "avg_time": round(sum(times) / len(times), 2),
            "avg_score": round(sum(scores) / len(scores), 2),
            "correlation": _pearson(tp),
        }

    return {
        "overall_correlation": _pearson(pairs),
        "total_pairs": len(pairs),
        "per_type": per_type,
    }


# =====================================================================
# Output formatting
# =====================================================================
def print_summary(
    records: list[EntityRecord],
    histogram: dict[str, int],
    clustering: dict[str, Any],
    per_type: dict[str, dict[str, Any]],
    dimension_variance: dict[str, dict[str, Any]],
    time_vs_score: dict[str, Any],
) -> None:
    """Print a human-readable summary of the analysis.

    Args:
        records: Entity records.
        histogram: Score histogram.
        clustering: Clustering metrics.
        per_type: Per-entity-type breakdown.
        dimension_variance: Dimension variance data.
        time_vs_score: Time vs score correlation data.
    """
    print("=" * 70)
    print("SCORE DISTRIBUTION ANALYSIS — Issue #294")
    print("=" * 70)
    print()

    # Overview
    print(f"Entities analyzed: {len(records)}")
    print(f"Total scores: {clustering['total_scores']}")
    print(f"Unique score values: {clustering['unique_scores']}")
    print(
        f"Score range: {clustering['min']}-{clustering['max']} "
        f"(spread: {clustering['score_range']})"
    )
    print(f"Mean: {clustering['mean']}  Std Dev: {clustering['std_dev']}  IQR: {clustering['iqr']}")
    print()

    # Histogram
    print("--- Score Histogram (0.5-wide bins) ---")
    max_count = max(histogram.values()) if histogram else 1
    for bin_label, count in histogram.items():
        bar = "#" * int(40 * count / max_count) if max_count > 0 else ""
        print(f"  {bin_label}: {count:>3} {bar}")
    print()

    # Modal scores (clustering indicator)
    print("--- Most Common Scores ---")
    for ms in clustering["modal_scores"]:
        pct = (
            round(100 * ms["count"] / clustering["total_scores"], 1)
            if clustering["total_scores"] > 0
            else 0
        )
        print(f"  {ms['score']:.1f}: {ms['count']} occurrences ({pct}%)")
    print()

    # Per-type breakdown
    print("--- Per-Entity-Type Breakdown ---")
    print(
        f"  {'Type':<12} {'Count':>5} {'Mean':>6} {'StdDev':>7} "
        f"{'Range':>10} {'AvgCreate':>10} {'AvgTotal':>10} {'MetRate':>8}"
    )
    print("  " + "-" * 80)
    for etype, stats in per_type.items():
        min_s = f"{stats['min_score']:.1f}" if stats["min_score"] is not None else "-"
        max_s = f"{stats['max_score']:.1f}" if stats["max_score"] is not None else "-"
        range_s = f"{min_s}-{max_s}"
        create_s = f"{stats['avg_creation_time']:.1f}s" if stats["avg_creation_time"] else "-"
        total_s = f"{stats['avg_total_time']:.1f}s" if stats["avg_total_time"] else "-"
        print(
            f"  {etype:<12} {stats['entity_count']:>5} {stats['mean_final_score']:>6.2f} "
            f"{stats['std_dev']:>7.3f} {range_s:>10} {create_s:>10} {total_s:>10} "
            f"{stats['threshold_met_rate']:>7.0%}"
        )
    print()

    # Dimension variance
    print("--- Per-Dimension Variance ---")
    for etype, data in dimension_variance.items():
        print(f"  {etype} (avg intra-entity spread: {data['avg_intra_entity_spread']:.1f}):")
        for dim, stats in data["dimensions"].items():
            print(
                f"    {dim:<25} mean={stats['mean']:.1f}  std={stats['std_dev']:.2f}  "
                f"range={stats['min']:.1f}-{stats['max']:.1f}  n={stats['count']}"
            )
    print()

    # Time vs score
    print("--- Generation Time vs Score ---")
    print(f"  Overall Pearson correlation: {time_vs_score['overall_correlation']:.3f}")
    for etype, data in time_vs_score["per_type"].items():
        print(
            f"  {etype:<12} avg_time={data['avg_time']:.1f}s  "
            f"avg_score={data['avg_score']:.1f}  r={data['correlation']:.3f}  n={data['count']}"
        )
    print()

    # Diagnostic conclusions
    print("=" * 70)
    print("DIAGNOSTIC CONCLUSIONS")
    print("=" * 70)

    if clustering["std_dev"] < 1.0:
        print("  [!] LOW SCORE SPREAD: std_dev < 1.0 — judge is clustering scores tightly")
    if clustering["iqr"] < 1.0:
        print("  [!] LOW IQR: interquartile range < 1.0 — most scores fall in a narrow band")

    # Check if fast entities score as well as slow ones
    fast_types = []
    slow_types = []
    for etype, data in time_vs_score["per_type"].items():
        if data["avg_time"] < 5.0:
            fast_types.append((etype, data["avg_score"]))
        elif data["avg_time"] > 8.0:
            slow_types.append((etype, data["avg_score"]))

    if fast_types and slow_types:
        fast_avg = sum(s for _, s in fast_types) / len(fast_types)
        slow_avg = sum(s for _, s in slow_types) / len(slow_types)
        diff = abs(fast_avg - slow_avg)
        if diff < 0.5:
            print(
                f"  [!] SCORE PARITY: fast entities ({fast_avg:.1f}) score similarly to "
                f"slow entities ({slow_avg:.1f}) — judge may not detect depth difference"
            )

    # Check intra-entity dimension spread
    for etype, data in dimension_variance.items():
        if data["avg_intra_entity_spread"] < 1.5:
            print(
                f"  [!] LOW DIMENSION SPREAD ({etype}): "
                f"avg intra-entity spread {data['avg_intra_entity_spread']:.1f} < 1.5 — "
                f"judge gives similar scores across all dimensions"
            )


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """Run the score distribution analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Score Distribution Analysis — diagnoses judge scoring bias from log files."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="output/logs/story_factory.log",
        help="Path to log file (default: output/logs/story_factory.log)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write JSON results (default: printed summary only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-entity records",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}")
        print("Run a world build first to generate scoring log data.")
        sys.exit(1)

    # Parse
    records = parse_log_file(log_path)
    if not records:
        print("No scoring data found in log file.")
        print("Ensure the world quality loop has been run and logging is at INFO level.")
        sys.exit(1)

    # Analyze
    histogram = compute_score_histogram(records)
    clustering = compute_clustering_metrics(records)
    per_type = compute_per_type_breakdown(records)
    dimension_variance = compute_dimension_variance(records)
    time_vs_score = compute_time_vs_score(records)

    # Print summary
    print_summary(records, histogram, clustering, per_type, dimension_variance, time_vs_score)

    # Verbose per-entity output
    if args.verbose:
        print()
        print("--- Per-Entity Details ---")
        for r in sorted(records, key=lambda x: (x.entity_type, x.name)):
            print(
                f"  [{r.entity_type}] {r.name}: scores={r.scores}, "
                f"create={r.creation_time:.1f}s, total={r.total_generation_time:.1f}s, "
                f"met_threshold={r.threshold_met}"
            )

    # Optional JSON output
    if args.output:
        output_data = {
            "log_file": str(log_path),
            "entity_count": len(records),
            "histogram": histogram,
            "clustering": clustering,
            "per_type": per_type,
            "dimension_variance": dimension_variance,
            "time_vs_score": time_vs_score,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
