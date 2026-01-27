"""Persistence mixin for StoryOrchestrator."""

import json
import logging
from datetime import datetime
from pathlib import Path

import src.settings as _settings
from src.memory.story_state import StoryState
from src.services.orchestrator._base import StoryOrchestratorBase

logger = logging.getLogger(__name__)


class PersistenceMixin(StoryOrchestratorBase):
    """Mixin providing persistence functionality."""

    def autosave(self) -> str | None:
        """Auto-save current state with timestamp update.

        Returns:
            The path where saved, or None if no story to save.
        """
        if not self.story_state:
            return None

        try:
            self.story_state.last_saved = datetime.now()
            self.story_state.updated_at = datetime.now()
            path = self.save_story()
            logger.debug(f"Autosaved story to {path}")
            return path
        except Exception as e:
            logger.warning(f"Autosave failed: {e}")
            return None

    def save_story(self, filepath: str | None = None) -> str:
        """Save the current story state to a JSON file.

        Args:
            filepath: Optional custom path. Defaults to output/stories/<story_id>.json

        Returns:
            The path where the story was saved.
        """
        if not self.story_state:
            raise ValueError("No story to save")

        # Update timestamps
        self.story_state.updated_at = datetime.now()
        if not self.story_state.last_saved:
            self.story_state.last_saved = datetime.now()

        # Default save location
        output_path: Path
        if not filepath:
            output_dir = _settings.STORIES_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.story_state.id}.json"
        else:
            output_path = Path(filepath)

        # Convert to dict for JSON serialization
        story_data = self.story_state.model_dump(mode="json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, default=str)

        logger.info(f"Story saved to {output_path}")
        return str(output_path)

    def load_story(self, filepath: str) -> StoryState:
        """Load a story state from a JSON file.

        Args:
            filepath: Path to the story JSON file.

        Returns:
            The loaded StoryState.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Story file not found: {path}")

        with open(path, encoding="utf-8") as f:
            story_data = json.load(f)

        self.story_state = StoryState.model_validate(story_data)
        # Set correlation ID for event tracking
        self._correlation_id = self.story_state.id[:8]
        logger.info(f"Story loaded from {path}")
        return self.story_state

    @staticmethod
    def list_saved_stories() -> list[dict[str, str | None]]:
        """List all saved stories in the output directory.

        Returns:
            List of dicts with story metadata (id, path, created_at, status, etc.)
        """
        output_dir = _settings.STORIES_DIR
        stories: list[dict[str, str | None]] = []

        if not output_dir.exists():
            return stories

        for filepath in output_dir.glob("*.json"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                stories.append(
                    {
                        "id": data.get("id"),
                        "path": str(filepath),
                        "created_at": data.get("created_at"),
                        "status": data.get("status"),
                        "premise": (
                            data.get("brief", {}).get("premise", "")[:100]
                            if data.get("brief")
                            else ""
                        ),
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not read story file {filepath}: {e}")

        return sorted(stories, key=lambda x: x.get("created_at") or "", reverse=True)
