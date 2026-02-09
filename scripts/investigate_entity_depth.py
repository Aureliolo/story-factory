#!/usr/bin/env python3
"""Entity Depth Analysis — measures content richness per entity type in world databases.

Reads SQLite world databases from output/worlds/ and compares entity content depth
across entity types (character, location, faction, item, concept):

1. Average text length per entity type (description + all text fields)
2. Field completeness — how many optional attributes are actually populated
3. Vocabulary diversity — unique word count / total words per entity type
4. Per-entity-type comparison to identify shallow entities

Usage:
    python scripts/investigate_entity_depth.py [options]
      --db-dir path/to/worlds   (default: output/worlds/)
      --db-file path/to/db      (analyze a single database)
      --output results.json     (default: printed summary only)
      --verbose                 (print per-entity details)
"""

import argparse
import json
import logging
import math
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fields expected per entity type for depth analysis.
# Based on Pydantic models in story_state.py, but excluding ``name`` and ``type``
# which are always-populated DB columns and don't indicate content depth.
EXPECTED_FIELDS: dict[str, list[str]] = {
    "character": [
        "role",
        "description",
        "personality_traits",
        "goals",
        "relationships",
        "arc_notes",
        "arc_type",
        "arc_progress",
    ],
    "faction": ["description", "leader", "goals", "values", "base_location"],
    "item": ["description", "significance", "properties"],
    "concept": ["description", "manifestations"],
    "location": ["description", "significance"],
}


# =====================================================================
# Data structures
# =====================================================================
class EntityDepthRecord:
    """Depth metrics for a single entity."""

    def __init__(
        self,
        entity_type: str,
        name: str,
        description: str,
        attributes: dict[str, Any],
    ) -> None:
        """Initialize an entity depth record.

        Args:
            entity_type: Entity type (e.g. "faction", "character").
            name: Entity display name.
            description: Entity description text.
            attributes: Additional entity attributes from the database.
        """
        self.entity_type = entity_type.lower()
        self.name = name
        self.description = description
        self.attributes = attributes

    @property
    def total_text(self) -> str:
        """Concatenate all text content from description and attributes."""
        parts = [self.description]
        for value in self.attributes.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.extend(str(v) for v in item.values() if isinstance(v, str))
            elif isinstance(value, dict):
                parts.extend(str(v) for v in value.values() if isinstance(v, str))
        return " ".join(parts)

    @property
    def total_text_length(self) -> int:
        """Total character count of all text content."""
        return len(self.total_text)

    @property
    def word_count(self) -> int:
        """Total word count across all text content."""
        return len(self.total_text.split())

    @property
    def vocabulary_diversity(self) -> float:
        """Ratio of unique words to total words (type-token ratio).

        Returns 0.0 if there are no words.
        """
        words = _normalize_words(self.total_text)
        if not words:
            return 0.0
        unique = len(set(words))
        return round(unique / len(words), 3)

    @property
    def populated_fields(self) -> int:
        """Count of non-empty attribute fields."""
        count = 0
        if self.description:
            count += 1
        for value in self.attributes.values():
            if _is_populated(value):
                count += 1
        return count

    @property
    def expected_field_count(self) -> int:
        """Number of fields expected for this entity type."""
        if self.entity_type in EXPECTED_FIELDS:
            return len(EXPECTED_FIELDS[self.entity_type])
        # Unknown entity type: fall back to description + attribute count
        logger.debug(
            "Unknown entity type %r — using actual field count as expected", self.entity_type
        )
        return 1 + len(self.attributes)

    @property
    def field_completeness(self) -> float:
        """Ratio of populated fields to expected fields (0.0-1.0)."""
        expected = self.expected_field_count
        if expected == 0:
            return 0.0
        return round(min(self.populated_fields / expected, 1.0), 3)


def _is_populated(value: Any) -> bool:
    """Check if a field value is meaningfully populated (not empty/default).

    Args:
        value: Field value to check.

    Returns:
        True if the value contains meaningful content.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    return True


def _normalize_words(text: str) -> list[str]:
    """Extract and normalize words from text for vocabulary analysis.

    Args:
        text: Raw text content.

    Returns:
        List of lowercase alphabetic words.
    """
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", text)]


# =====================================================================
# Database reading
# =====================================================================
def read_world_database(db_path: Path) -> list[EntityDepthRecord]:
    """Read all entities from a world SQLite database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of EntityDepthRecord objects.
    """
    logger.info("Reading world database: %s", db_path)

    records: list[EntityDepthRecord] = []

    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check if entities table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entities'")
            if not cursor.fetchone():
                logger.warning("No 'entities' table in %s", db_path)
                return []

            cursor.execute(
                "SELECT type, name, description, attributes FROM entities ORDER BY type, name"
            )
            for row in cursor.fetchall():
                entity_type = row["type"]
                name = row["name"]
                description = row["description"] or ""
                try:
                    raw = json.loads(row["attributes"]) if row["attributes"] else {}
                    attributes = raw if isinstance(raw, dict) else {}
                except json.JSONDecodeError, TypeError:
                    attributes = {}

                records.append(EntityDepthRecord(entity_type, name, description, attributes))
    except sqlite3.Error as e:
        logger.error("Failed to read database %s: %s", db_path, e)

    logger.info("Read %d entities from %s", len(records), db_path)
    return records


def find_world_databases(db_dir: Path) -> list[Path]:
    """Find all SQLite world database files in a directory.

    Args:
        db_dir: Directory to search.

    Returns:
        List of paths to .db or .sqlite files.
    """
    dbs: list[Path] = []
    if not db_dir.exists():
        logger.debug("Database directory does not exist: %s", db_dir)
        return dbs
    if not db_dir.is_dir():
        logger.warning("Database path exists but is not a directory: %s", db_dir)
        return dbs

    # Look for .db and .sqlite files
    for pattern in ("*.db", "*.sqlite", "*.sqlite3"):
        dbs.extend(db_dir.glob(pattern))

    # Also check if any files without extension are SQLite databases
    for f in db_dir.iterdir():
        if f.is_file() and f.suffix == "" and f not in dbs:
            try:
                # Check SQLite magic bytes
                with open(f, "rb") as fh:
                    header = fh.read(16)
                    if header.startswith(b"SQLite format 3"):
                        dbs.append(f)
            except OSError as e:
                logger.debug("Could not read file %s for SQLite detection: %s", f, e)

    return sorted(dbs)


# =====================================================================
# Analysis functions
# =====================================================================
def compute_type_depth_stats(records: list[EntityDepthRecord]) -> dict[str, dict[str, Any]]:
    """Compute depth statistics per entity type.

    Args:
        records: Entity depth records.

    Returns:
        Dict mapping entity type to depth statistics.
    """
    type_groups: dict[str, list[EntityDepthRecord]] = defaultdict(list)
    for r in records:
        type_groups[r.entity_type].append(r)

    def _stats(values: list[float]) -> dict[str, float]:
        """Compute mean, std_dev, min, max for a list of numeric values."""
        if not values:
            return {"mean": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0}
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return {
            "mean": round(mean, 2),
            "std_dev": round(math.sqrt(variance), 3),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    result: dict[str, dict[str, Any]] = {}
    for entity_type, group in sorted(type_groups.items()):
        text_lengths = [r.total_text_length for r in group]
        word_counts = [r.word_count for r in group]
        vocab_diversities = [r.vocabulary_diversity for r in group]
        completeness_vals = [r.field_completeness for r in group]

        result[entity_type] = {
            "entity_count": len(group),
            "text_length": _stats([float(v) for v in text_lengths]),
            "word_count": _stats([float(v) for v in word_counts]),
            "vocabulary_diversity": _stats(vocab_diversities),
            "field_completeness": _stats(completeness_vals),
        }

    return result


def compute_cross_type_comparison(
    type_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare depth metrics across entity types to identify imbalances.

    Args:
        type_stats: Per-type depth statistics.

    Returns:
        Comparison summary with rankings and imbalance flags.
    """
    if not type_stats:
        return {"rankings": {}, "imbalances": []}

    # Rank by average word count
    word_count_ranking = sorted(
        type_stats.items(),
        key=lambda x: x[1]["word_count"]["mean"],
        reverse=True,
    )

    # Rank by vocabulary diversity
    vocab_ranking = sorted(
        type_stats.items(),
        key=lambda x: x[1]["vocabulary_diversity"]["mean"],
        reverse=True,
    )

    # Rank by field completeness
    completeness_ranking = sorted(
        type_stats.items(),
        key=lambda x: x[1]["field_completeness"]["mean"],
        reverse=True,
    )

    rankings = {
        "by_word_count": [
            {"type": t, "mean_words": s["word_count"]["mean"]} for t, s in word_count_ranking
        ],
        "by_vocabulary_diversity": [
            {"type": t, "mean_diversity": s["vocabulary_diversity"]["mean"]}
            for t, s in vocab_ranking
        ],
        "by_field_completeness": [
            {"type": t, "mean_completeness": s["field_completeness"]["mean"]}
            for t, s in completeness_ranking
        ],
    }

    # Detect imbalances (>2x difference between best and worst)
    imbalances: list[str] = []
    if word_count_ranking:
        best_words = word_count_ranking[0][1]["word_count"]["mean"]
        worst_words = word_count_ranking[-1][1]["word_count"]["mean"]
        if worst_words > 0 and best_words / worst_words > 2.0:
            imbalances.append(
                f"Word count imbalance: {word_count_ranking[0][0]} "
                f"({best_words:.0f} words) vs {word_count_ranking[-1][0]} "
                f"({worst_words:.0f} words) — {best_words / worst_words:.1f}x difference"
            )

    return {"rankings": rankings, "imbalances": imbalances}


# =====================================================================
# Output formatting
# =====================================================================
def print_summary(
    records: list[EntityDepthRecord],
    type_stats: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    db_sources: list[str],
) -> None:
    """Print a human-readable summary of entity depth analysis.

    Args:
        records: All entity records analyzed.
        type_stats: Per-type depth statistics.
        comparison: Cross-type comparison results.
        db_sources: List of database file paths analyzed.
    """
    print("=" * 70)
    print("ENTITY DEPTH ANALYSIS — Issue #294")
    print("=" * 70)
    print()
    print(f"Databases analyzed: {len(db_sources)}")
    for db in db_sources:
        print(f"  - {db}")
    print(f"Total entities: {len(records)}")
    print()

    # Per-type summary
    print("--- Per-Entity-Type Depth ---")
    print(
        f"  {'Type':<12} {'Count':>5} {'AvgWords':>9} {'AvgChars':>9} "
        f"{'VocabDiv':>9} {'Complete':>9}"
    )
    print("  " + "-" * 62)
    for etype, stats in sorted(type_stats.items()):
        print(
            f"  {etype:<12} {stats['entity_count']:>5} "
            f"{stats['word_count']['mean']:>9.0f} "
            f"{stats['text_length']['mean']:>9.0f} "
            f"{stats['vocabulary_diversity']['mean']:>9.3f} "
            f"{stats['field_completeness']['mean']:>8.0%}"
        )
    print()

    # Rankings
    print("--- Rankings ---")
    for metric, items in comparison["rankings"].items():
        display_metric = metric.replace("by_", "").replace("_", " ").title()
        ranking_str = " > ".join(f"{item['type']}" for item in items)
        print(f"  {display_metric}: {ranking_str}")
    print()

    # Imbalances
    if comparison["imbalances"]:
        print("--- Detected Imbalances ---")
        for imbalance in comparison["imbalances"]:
            print(f"  [!] {imbalance}")
    else:
        print("--- No significant depth imbalances detected ---")
    print()

    # Diagnostic conclusions
    print("=" * 70)
    print("DIAGNOSTIC CONCLUSIONS")
    print("=" * 70)

    # Check for shallow entity types
    for etype, stats in type_stats.items():
        avg_words = stats["word_count"]["mean"]
        completeness = stats["field_completeness"]["mean"]
        if avg_words < 50:
            print(f"  [!] SHALLOW: {etype} averages only {avg_words:.0f} words per entity")
        if completeness < 0.5:
            print(f"  [!] INCOMPLETE: {etype} has only {completeness:.0%} field completeness")

    # Check vocabulary diversity
    for etype, stats in type_stats.items():
        vocab_div = stats["vocabulary_diversity"]["mean"]
        if vocab_div < 0.4:
            print(
                f"  [!] LOW VOCABULARY DIVERSITY: {etype} has TTR of {vocab_div:.3f} "
                f"— may indicate repetitive/generic content"
            )


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """Run the entity depth analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Entity Depth Analysis — measures content richness per entity type."
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="output/worlds",
        help="Directory containing world databases (default: output/worlds/)",
    )
    parser.add_argument(
        "--db-file",
        type=str,
        help="Analyze a single database file instead of a directory",
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

    # Find databases
    if args.db_file:
        db_paths = [Path(args.db_file)]
        if not db_paths[0].exists():
            print(f"ERROR: Database file not found: {args.db_file}")
            sys.exit(1)
    else:
        db_dir = Path(args.db_dir)
        db_paths = find_world_databases(db_dir)
        if not db_paths:
            print(f"ERROR: No world databases found in {db_dir}")
            print("Run a world build first to generate entity data.")
            sys.exit(1)

    # Read all databases
    all_records: list[EntityDepthRecord] = []
    db_sources: list[str] = []
    for db_path in db_paths:
        records = read_world_database(db_path)
        if records:
            all_records.extend(records)
            db_sources.append(str(db_path))

    if not all_records:
        print("No entities found in any database.")
        sys.exit(1)

    # Analyze
    type_stats = compute_type_depth_stats(all_records)
    comparison = compute_cross_type_comparison(type_stats)

    # Print summary
    print_summary(all_records, type_stats, comparison, db_sources)

    # Verbose per-entity output
    if args.verbose:
        print()
        print("--- Per-Entity Details ---")
        for r in sorted(all_records, key=lambda x: (x.entity_type, x.name)):
            print(
                f"  [{r.entity_type}] {r.name}: "
                f"words={r.word_count}, chars={r.total_text_length}, "
                f"vocab_div={r.vocabulary_diversity:.3f}, "
                f"completeness={r.field_completeness:.0%}, "
                f"populated={r.populated_fields}/{r.expected_field_count}"
            )

    # Optional JSON output
    if args.output:
        output_data = {
            "databases": db_sources,
            "entity_count": len(all_records),
            "type_stats": type_stats,
            "comparison": comparison,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
