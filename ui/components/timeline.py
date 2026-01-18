"""Timeline visualization component using NiceGUI native components.

A chapter-based Gantt-style timeline showing story elements across chapters.
"""

import logging
from collections.abc import Callable

from nicegui import ui

from memory.story_state import StoryState
from settings import Settings
from utils.json_parser import clean_llm_text

logger = logging.getLogger(__name__)


# Color palette for timeline tracks
TRACK_COLORS = {
    "events": {"bg": "#3b82f6", "border": "#2563eb", "name": "Blue"},
    "chapters": {"bg": "#10b981", "border": "#059669", "name": "Green"},
    "characters": {"bg": "#f59e0b", "border": "#d97706", "name": "Orange"},
    "locations": {"bg": "#8b5cf6", "border": "#7c3aed", "name": "Purple"},
}


def _get_character_spans(story_state: StoryState) -> list[dict]:
    """Get chapter spans for each character.

    Args:
        story_state: Story state to extract character data from.

    Returns:
        List of span dictionaries with name, start, end, and color.
    """
    spans = []
    for char in story_state.characters:
        # Find first and last chapter where character appears
        if char.arc_progress:
            chapters = list(char.arc_progress.keys())
            first_ch = min(chapters)
            last_ch = max(chapters)
        else:
            # Default to chapter 1 if no arc progress
            first_ch = 1
            last_ch = 1

        spans.append(
            {
                "name": char.name,
                "role": char.role,
                "start": first_ch,
                "end": last_ch,
                "color": TRACK_COLORS["characters"]["bg"],
            }
        )

    logger.debug(f"Extracted {len(spans)} character spans")
    return spans


def _get_event_spans(story_state: StoryState) -> list[dict]:
    """Get chapter spans for timeline events.

    Args:
        story_state: Story state to extract event data from.

    Returns:
        List of span dictionaries with name, chapter, and color.
    """
    spans = []
    for idx, event in enumerate(story_state.timeline):
        # Events are point-in-time, assign to chapter index + 1
        chapter = min(idx + 1, max(1, len(story_state.chapters)))

        spans.append(
            {
                "name": event[:50] + "..." if len(event) > 50 else event,
                "full_text": event,
                "chapter": chapter,
                "color": TRACK_COLORS["events"]["bg"],
            }
        )

    logger.debug(f"Extracted {len(spans)} event spans")
    return spans


def _get_chapter_spans(story_state: StoryState) -> list[dict]:
    """Get chapter data for visualization.

    Args:
        story_state: Story state to extract chapter data from.

    Returns:
        List of chapter dictionaries with number, title, and color.
    """
    spans = []
    for chapter in story_state.chapters:
        spans.append(
            {
                "number": chapter.number,
                "title": chapter.title,
                "word_count": chapter.word_count,
                "color": TRACK_COLORS["chapters"]["bg"],
            }
        )

    logger.debug(f"Extracted {len(spans)} chapter spans")
    return spans


def _get_location_spans(story_state: StoryState) -> list[dict]:
    """Get location spans across chapters.

    Args:
        story_state: Story state to extract location data from.

    Returns:
        List of span dictionaries with name, start, end, and color.
    """
    spans = []

    # Use world description as primary location if available
    if story_state.world_description:
        # Clean the description to remove LLM artifacts like <think> tags
        clean_desc = clean_llm_text(story_state.world_description)
        spans.append(
            {
                "name": "Primary Setting",
                "description": clean_desc[:100] + "..." if len(clean_desc) > 100 else clean_desc,
                "start": 1,
                "end": max(1, len(story_state.chapters)),
                "color": TRACK_COLORS["locations"]["bg"],
            }
        )

    # Extract unique locations from brief if available
    if story_state.brief and story_state.brief.setting_place:
        existing_name = str(spans[0].get("name", "")) if spans else ""
        if not spans or story_state.brief.setting_place not in existing_name:
            spans.append(
                {
                    "name": story_state.brief.setting_place,
                    "description": f"Time: {story_state.brief.setting_time}"
                    if story_state.brief.setting_time
                    else "",
                    "start": 1,
                    "end": max(1, len(story_state.chapters)),
                    "color": TRACK_COLORS["locations"]["bg"],
                }
            )

    logger.debug(f"Extracted {len(spans)} location spans")
    return spans


class TimelineComponent:
    """Interactive timeline visualization for story events.

    A chapter-based Gantt-style chart showing:
    - X-axis: Chapter numbers (Ch. 1, Ch. 2, Ch. 3...)
    - Y-axis tracks: Story Events, Chapters, Characters, Locations
    - Items span from first appearance to last appearance

    Features:
    - Native NiceGUI components (no external JS library)
    - Full dark mode support
    - Hover tooltips with full details
    - Responsive grid that scales with chapter count
    """

    def __init__(
        self,
        story_state: StoryState | None = None,
        settings: Settings | None = None,
        on_item_click: Callable[[str, dict], None] | None = None,
        height: int = 400,
        editable: bool = True,
    ):
        """Initialize timeline component.

        Args:
            story_state: StoryState to visualize.
            settings: Application settings for configuration values.
            on_item_click: Callback when an item is clicked (receives track_name, item_data).
            height: Timeline container height in pixels.
            editable: Whether items can be edited (reserved for future use).
        """
        from settings import Settings

        self.story_state = story_state
        self.settings = settings or Settings.load()
        self.on_item_click = on_item_click
        self.height = height
        self.editable = editable

        self._container: ui.column | None = None
        logger.debug("TimelineComponent initialized")

    def build(self) -> None:
        """Build the timeline UI."""
        with ui.column().classes("w-full"):
            # Controls
            with ui.row().classes("w-full items-center gap-4 mb-2"):
                ui.label("Story Timeline").classes("text-lg font-semibold")
                ui.space()

                # Legend
                with ui.row().classes("gap-2"):
                    for track_id, colors in TRACK_COLORS.items():
                        with ui.row().classes("items-center gap-1"):
                            ui.html(
                                f'<div style="width: 12px; height: 12px; background: {colors["bg"]}; '
                                f'border-radius: 2px;"></div>',
                                sanitize=False,
                            )
                            ui.label(track_id.title()).classes(
                                "text-xs text-gray-600 dark:text-gray-400"
                            )

                # Refresh button
                ui.button(
                    icon="refresh",
                    on_click=self.refresh,
                ).props("flat round dense").tooltip("Refresh timeline")

            # Timeline container
            self._container = ui.column().classes(
                "w-full border border-gray-300 dark:border-gray-700 rounded-lg overflow-x-auto "
                "bg-gray-50 dark:bg-gray-900"
            )

            # Initial render
            self._render_timeline()

    def set_story_state(self, story_state: StoryState | None) -> None:
        """Set the story state and refresh.

        Args:
            story_state: StoryState to visualize.
        """
        logger.debug(f"Setting story state: {story_state.id if story_state else None}")
        self.story_state = story_state
        self._render_timeline()

    def refresh(self) -> None:
        """Refresh the timeline visualization."""
        logger.debug("Refreshing timeline")
        self._render_timeline()

    def _render_timeline(self) -> None:
        """Render the timeline content."""
        if not self._container:
            logger.warning("Timeline container not initialized")
            return

        self._container.clear()

        if not self.story_state:
            self._render_empty_state()
            return

        # Check if there's meaningful data to display
        has_data = (
            self.story_state.chapters or self.story_state.timeline or self.story_state.characters
        )

        if not has_data:
            self._render_empty_state()
            return

        self._build_chapter_timeline()

    def _render_empty_state(self) -> None:
        """Render empty state message."""
        assert self._container is not None  # Checked by caller
        with self._container:
            with (
                ui.column()
                .classes("w-full items-center justify-center gap-4 p-8")
                .style(f"min-height: {self.height}px")
            ):
                ui.icon("timeline", size="xl").classes("text-gray-400 dark:text-gray-500")
                ui.label("No timeline data available").classes("text-gray-500 dark:text-gray-400")
                ui.label(
                    "Complete the interview and generate chapters to see your story timeline."
                ).classes("text-sm text-gray-400 dark:text-gray-500 text-center")

    def _build_chapter_timeline(self) -> None:
        """Build chapter-based timeline using native NiceGUI."""
        assert self.story_state is not None  # Checked by caller
        assert self._container is not None  # Checked by caller
        chapters = self.story_state.chapters
        num_chapters = max(1, len(chapters))

        # Calculate grid column widths
        label_width = "120px"
        chapter_width = "minmax(80px, 1fr)"

        with self._container:
            with ui.column().classes("w-full p-4 gap-2").style(f"min-height: {self.height}px"):
                # Header row with chapter numbers
                self._build_header_row(num_chapters, label_width, chapter_width)

                # Track rows
                self._build_chapters_track(chapters, label_width, chapter_width)
                self._build_events_track(num_chapters, label_width, chapter_width)
                self._build_characters_track(num_chapters, label_width, chapter_width)
                self._build_locations_track(num_chapters, label_width, chapter_width)

    def _build_header_row(self, num_chapters: int, label_width: str, chapter_width: str) -> None:
        """Build the header row with chapter numbers.

        Args:
            num_chapters: Number of chapters to display.
            label_width: Width of the track label column.
            chapter_width: Width of each chapter column.
        """
        grid_template = f"{label_width} " + " ".join([chapter_width] * num_chapters)

        with (
            ui.row()
            .classes("w-full gap-0")
            .style(f"display: grid; grid-template-columns: {grid_template}; align-items: center;")
        ):
            # Empty cell for track label column
            ui.label("").classes("text-sm font-semibold")

            # Chapter header cells
            for i in range(1, num_chapters + 1):
                ui.label(f"Ch. {i}").classes(
                    "text-center text-sm font-semibold text-gray-700 dark:text-gray-300 "
                    "border-l dark:border-gray-600 py-2"
                )

    def _build_chapters_track(self, chapters: list, label_width: str, chapter_width: str) -> None:
        """Build the chapters track row.

        Args:
            chapters: List of chapter objects.
            label_width: Width of the track label column.
            chapter_width: Width of each chapter column.
        """
        num_chapters = max(1, len(chapters))
        grid_template = f"{label_width} " + " ".join([chapter_width] * num_chapters)

        with (
            ui.row()
            .classes("w-full gap-0")
            .style(
                f"display: grid; grid-template-columns: {grid_template}; "
                "align-items: stretch; min-height: 50px;"
            )
        ):
            # Track label
            ui.label("Chapters").classes(
                "text-sm font-medium text-gray-600 dark:text-gray-400 flex items-center"
            )

            # Chapter cells
            for chapter in chapters:
                self._build_chapter_cell(chapter)

            # Fill remaining cells if chapters list is shorter
            for _ in range(num_chapters - len(chapters)):
                ui.html('<div class="h-full border-l dark:border-gray-600"></div>', sanitize=False)

    def _build_chapter_cell(self, chapter) -> None:
        """Build a single chapter cell.

        Args:
            chapter: Chapter object to display.
        """
        color = TRACK_COLORS["chapters"]["bg"]
        border_color = TRACK_COLORS["chapters"]["border"]

        with ui.element("div").classes(
            "h-full border-l dark:border-gray-600 p-1 flex items-center"
        ):
            with (
                ui.element("div")
                .classes(
                    "w-full px-2 py-1 rounded text-white text-xs truncate cursor-pointer "
                    "hover:opacity-80 transition-opacity"
                )
                .style(f"background-color: {color}; border: 1px solid {border_color};")
                .tooltip(f"Chapter {chapter.number}: {chapter.title}\n{chapter.word_count:,} words")
            ):
                ui.label(chapter.title[:20] + "..." if len(chapter.title) > 20 else chapter.title)

    def _build_events_track(self, num_chapters: int, label_width: str, chapter_width: str) -> None:
        """Build the events track row.

        Args:
            num_chapters: Number of chapter columns.
            label_width: Width of the track label column.
            chapter_width: Width of each chapter column.
        """
        assert self.story_state is not None  # Checked by caller
        grid_template = f"{label_width} " + " ".join([chapter_width] * num_chapters)
        events = _get_event_spans(self.story_state)

        # Group events by chapter
        events_by_chapter: dict[int, list] = {i: [] for i in range(1, num_chapters + 1)}
        for event in events:
            ch = event["chapter"]
            if ch in events_by_chapter:
                events_by_chapter[ch].append(event)

        with (
            ui.row()
            .classes("w-full gap-0")
            .style(
                f"display: grid; grid-template-columns: {grid_template}; "
                "align-items: stretch; min-height: 50px;"
            )
        ):
            # Track label
            ui.label("Events").classes(
                "text-sm font-medium text-gray-600 dark:text-gray-400 flex items-center"
            )

            # Event cells per chapter
            for i in range(1, num_chapters + 1):
                self._build_events_cell(events_by_chapter.get(i, []))

    def _build_events_cell(self, events: list) -> None:
        """Build a cell containing events.

        Args:
            events: List of event dictionaries for this chapter.
        """
        color = TRACK_COLORS["events"]["bg"]
        border_color = TRACK_COLORS["events"]["border"]

        with ui.element("div").classes(
            "h-full border-l dark:border-gray-600 p-1 flex flex-col gap-1"
        ):
            if not events:
                ui.html('<div class="flex-1"></div>', sanitize=False)
            else:
                for event in events[:3]:  # Limit to 3 events per cell
                    with (
                        ui.element("div")
                        .classes(
                            "w-full px-2 py-1 rounded text-white text-xs truncate "
                            "cursor-pointer hover:opacity-80 transition-opacity"
                        )
                        .style(f"background-color: {color}; border: 1px solid {border_color};")
                        .tooltip(event.get("full_text", event["name"]))
                    ):
                        ui.label(
                            event["name"][:15] + "..." if len(event["name"]) > 15 else event["name"]
                        )

                if len(events) > 3:
                    ui.label(f"+{len(events) - 3} more").classes(
                        "text-xs text-gray-500 dark:text-gray-400"
                    )

    def _build_characters_track(
        self, num_chapters: int, label_width: str, chapter_width: str
    ) -> None:
        """Build the characters track row.

        Args:
            num_chapters: Number of chapter columns.
            label_width: Width of the track label column.
            chapter_width: Width of each chapter column.
        """
        assert self.story_state is not None  # Checked by caller
        grid_template = f"{label_width} " + " ".join([chapter_width] * num_chapters)
        characters = _get_character_spans(self.story_state)

        with (
            ui.row()
            .classes("w-full gap-0")
            .style(
                f"display: grid; grid-template-columns: {grid_template}; "
                "align-items: stretch; min-height: 60px;"
            )
        ):
            # Track label
            ui.label("Characters").classes(
                "text-sm font-medium text-gray-600 dark:text-gray-400 flex items-center"
            )

            # Character spans across chapters
            self._build_character_spans(characters, num_chapters)

    def _build_character_spans(self, characters: list, num_chapters: int) -> None:
        """Build character span bars across chapters.

        Args:
            characters: List of character span dictionaries.
            num_chapters: Total number of chapters.
        """
        color = TRACK_COLORS["characters"]["bg"]
        border_color = TRACK_COLORS["characters"]["border"]

        # Create a container for the full chapter span area
        with ui.element("div").classes("contents"):
            for i in range(1, num_chapters + 1):
                # Find characters that span this chapter
                chars_in_chapter = [c for c in characters if c["start"] <= i <= c["end"]]

                with ui.element("div").classes(
                    "h-full border-l dark:border-gray-600 p-1 flex flex-col gap-1"
                ):
                    if not chars_in_chapter:
                        ui.html('<div class="flex-1"></div>', sanitize=False)
                    else:
                        for char in chars_in_chapter[:4]:  # Limit display
                            # Show full bar for characters that start here
                            is_start = char["start"] == i
                            is_end = char["end"] == i

                            style = f"background-color: {color}; border: 1px solid {border_color};"
                            if not is_start:
                                style += " border-left: none; border-top-left-radius: 0; border-bottom-left-radius: 0;"
                            if not is_end:
                                style += " border-right: none; border-top-right-radius: 0; border-bottom-right-radius: 0;"

                            with (
                                ui.element("div")
                                .classes(
                                    "w-full px-1 py-0.5 text-white text-xs truncate "
                                    "cursor-pointer hover:opacity-80 transition-opacity rounded"
                                )
                                .style(style)
                                .tooltip(
                                    f"{char['name']} ({char['role']})\nChapters {char['start']}-{char['end']}"
                                )
                            ):
                                if is_start:
                                    ui.label(
                                        char["name"][:10] + "..."
                                        if len(char["name"]) > 10
                                        else char["name"]
                                    )
                                else:
                                    ui.label("")  # Continue bar without repeating name

                        if len(chars_in_chapter) > 4:
                            ui.label(f"+{len(chars_in_chapter) - 4}").classes(
                                "text-xs text-gray-500 dark:text-gray-400"
                            )

    def _build_locations_track(
        self, num_chapters: int, label_width: str, chapter_width: str
    ) -> None:
        """Build the locations track row.

        Args:
            num_chapters: Number of chapter columns.
            label_width: Width of the track label column.
            chapter_width: Width of each chapter column.
        """
        assert self.story_state is not None  # Checked by caller
        grid_template = f"{label_width} " + " ".join([chapter_width] * num_chapters)
        locations = _get_location_spans(self.story_state)

        with (
            ui.row()
            .classes("w-full gap-0")
            .style(
                f"display: grid; grid-template-columns: {grid_template}; "
                "align-items: stretch; min-height: 50px;"
            )
        ):
            # Track label
            ui.label("Locations").classes(
                "text-sm font-medium text-gray-600 dark:text-gray-400 flex items-center"
            )

            # Location spans
            self._build_location_spans(locations, num_chapters)

    def _build_location_spans(self, locations: list, num_chapters: int) -> None:
        """Build location span bars across chapters.

        Args:
            locations: List of location span dictionaries.
            num_chapters: Total number of chapters.
        """
        color = TRACK_COLORS["locations"]["bg"]
        border_color = TRACK_COLORS["locations"]["border"]

        for i in range(1, num_chapters + 1):
            # Find locations that span this chapter
            locs_in_chapter = [loc for loc in locations if loc["start"] <= i <= loc["end"]]

            with ui.element("div").classes(
                "h-full border-l dark:border-gray-600 p-1 flex flex-col gap-1"
            ):
                if not locs_in_chapter:
                    ui.html('<div class="flex-1"></div>', sanitize=False)
                else:
                    for loc in locs_in_chapter[:2]:  # Limit display
                        is_start = loc["start"] == i
                        is_end = loc["end"] == i

                        style = f"background-color: {color}; border: 1px solid {border_color};"
                        if not is_start:
                            style += " border-left: none; border-top-left-radius: 0; border-bottom-left-radius: 0;"
                        if not is_end:
                            style += " border-right: none; border-top-right-radius: 0; border-bottom-right-radius: 0;"

                        with (
                            ui.element("div")
                            .classes(
                                "w-full px-1 py-0.5 text-white text-xs truncate "
                                "cursor-pointer hover:opacity-80 transition-opacity rounded"
                            )
                            .style(style)
                            .tooltip(f"{loc['name']}\n{loc.get('description', '')}")
                        ):
                            if is_start:
                                ui.label(
                                    loc["name"][:15] + "..."
                                    if len(loc["name"]) > 15
                                    else loc["name"]
                                )
                            else:
                                ui.label("")  # Continue bar without repeating name
