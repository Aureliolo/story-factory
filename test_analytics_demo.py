#!/usr/bin/env python3
"""Demo script to populate analytics database with sample data for testing."""

import logging
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from memory.mode_database import ModeDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def populate_sample_data():
    """Populate database with sample analytics data."""
    # Use a test database
    db_path = Path("output/test_analytics.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing test database
    if db_path.exists():
        db_path.unlink()

    db = ModeDatabase(db_path)

    logger.info("Populating sample analytics data...")

    # Sample models
    models = [
        "llama3.2:3b",
        "qwen2.5:7b",
        "mistral:7b",
        "phi3.5:3.8b",
    ]

    # Sample roles
    roles = ["writer", "editor", "interviewer", "architect"]

    # Sample genres
    genres = ["fantasy", "sci-fi", "mystery", "romance"]

    # Generate sample scores over the past 30 days
    base_time = datetime.now() - timedelta(days=30)

    score_id_counter = 1
    for day in range(30):
        current_time = base_time + timedelta(days=day)

        # Generate 5-10 scores per day
        num_scores = random.randint(5, 10)

        for _ in range(num_scores):
            model = random.choice(models)
            role = random.choice(roles)
            genre = random.choice(genres)

            # Quality scores tend to improve over time (simulate learning)
            time_factor = day / 30.0
            base_quality = 6.0 + (time_factor * 2.0)

            prose_quality = base_quality + random.uniform(-1.0, 1.5)
            instruction_following = base_quality + random.uniform(-0.5, 1.0)
            consistency_score = base_quality + random.uniform(-1.0, 1.0)

            # Clamp to 0-10 range
            prose_quality = max(0, min(10, prose_quality))
            instruction_following = max(0, min(10, instruction_following))
            consistency_score = max(0, min(10, consistency_score))

            # Performance metrics
            tokens_generated = random.randint(500, 2000)
            time_seconds = random.uniform(10, 60)
            tokens_per_second = tokens_generated / time_seconds

            # Insert with custom timestamp using direct SQL for demo data
            conn = sqlite3.connect(db.db_path)
            conn.execute(
                """
                INSERT INTO generation_scores (
                    timestamp, project_id, chapter_id, agent_role, model_id, mode_name, genre,
                    tokens_generated, time_seconds, tokens_per_second,
                    prose_quality, instruction_following, consistency_score,
                    was_regenerated, edit_distance, user_rating
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current_time.isoformat(),
                    f"project_{random.randint(1, 5)}",
                    f"chapter_{random.randint(1, 20)}",
                    role,
                    model,
                    "balanced",
                    genre,
                    tokens_generated,
                    time_seconds,
                    tokens_per_second,
                    prose_quality,
                    instruction_following,
                    consistency_score,
                    random.choice([0, 0, 0, 1]),  # 25% regeneration rate
                    random.randint(0, 100) if random.random() < 0.3 else None,
                    random.randint(1, 5) if random.random() < 0.2 else None,
                ),
            )
            conn.commit()
            conn.close()

            score_id_counter += 1

    logger.info(f"✓ Created {score_id_counter - 1} generation scores")

    # Update model performance aggregates
    for model in models:
        for role in roles:
            db.update_model_performance(model, role)

    logger.info("✓ Updated model performance aggregates")

    # Add some world entity scores
    entity_types = ["character", "location", "faction", "item"]
    for i in range(50):
        entity_type = random.choice(entity_types)
        scores_dict = {
            "depth": random.uniform(7, 10),
            "consistency": random.uniform(7, 10),
            "originality": random.uniform(6, 10),
            "detail": random.uniform(7, 10),
        }
        avg_score = sum(scores_dict.values()) / len(scores_dict)
        scores_dict["average"] = avg_score

        db.record_world_entity_score(
            project_id=f"project_{random.randint(1, 3)}",
            entity_type=entity_type,
            entity_name=f"Test_{entity_type}_{i}",
            model_id=random.choice(models),
            scores=scores_dict,
            iterations_used=random.randint(1, 3),
            generation_time_seconds=random.uniform(5, 20),
        )

    logger.info("✓ Created 50 world entity scores")

    # Add some recommendations
    rec_types = ["model_swap", "temperature_adjust", "prompt_refinement"]
    for i in range(10):
        db.record_recommendation(
            recommendation_type=random.choice(rec_types),
            current_value=f"model_{i}",
            suggested_value=f"better_model_{i}",
            reason=f"Performance data suggests this change would improve quality by ~{random.randint(5, 15)}%",
            confidence=random.uniform(0.7, 0.95),
            affected_role=random.choice(roles),
            evidence={"sample_size": random.randint(10, 50)},
            expected_improvement=f"+{random.uniform(0.5, 1.5):.1f} quality points",
        )

    logger.info("✓ Created 10 recommendations")

    logger.info(f"\n✅ Sample data populated successfully in {db_path}")
    logger.info(f"   Total scores: {db.get_score_count()}")
    logger.info(f"   Unique genres: {len(db.get_unique_genres())}")

    return str(db_path)


if __name__ == "__main__":
    db_path = populate_sample_data()
    print(f"\nTo use this database, set DB_PATH={db_path} or copy to output/model_scores.db")
