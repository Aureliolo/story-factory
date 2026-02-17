"""Entity card component for displaying world entities."""

from collections.abc import Callable
from typing import Any

from nicegui import ui

from src.memory.entities import Entity
from src.ui.theme import get_entity_icon, get_role_border_style
from src.utils.constants import get_entity_color


def _get_quality_badge_color(score: float) -> str:
    """Get badge color based on quality score.

    Args:
        score: Quality score (0-10).

    Returns:
        Color string for the badge.
    """
    if score >= 8.0:
        return "green"
    elif score >= 6.0:
        return "orange"
    else:
        return "red"


def _format_quality_tooltip(quality_scores: dict[str, Any]) -> str:
    """Format quality scores for tooltip display.

    Args:
        quality_scores: Dictionary of quality dimension scores.

    Returns:
        Formatted tooltip string.
    """
    lines = []
    for key, value in quality_scores.items():
        if key == "feedback":
            continue  # Skip feedback in tooltip
        if isinstance(value, (int, float)):
            lines.append(f"{key.replace('_', ' ').title()}: {value:.1f}")
    return "\n".join(lines)


class EntityCard:
    """Card component for displaying an entity.

    Features:
    - Type-colored header
    - Name and description
    - Attributes display
    - Edit/delete actions
    """

    def __init__(
        self,
        entity: Entity,
        on_edit: Callable[[Entity], Any] | None = None,
        on_delete: Callable[[Entity], Any] | None = None,
        on_select: Callable[[Entity], Any] | None = None,
        selected: bool = False,
        compact: bool = False,
    ):
        """Initialize entity card.

        Args:
            entity: Entity to display.
            on_edit: Callback when edit is clicked.
            on_delete: Callback when delete is clicked.
            on_select: Callback when card is clicked.
            selected: Whether this card is selected.
            compact: Whether to use compact display mode.
        """
        self.entity = entity
        self.on_edit = on_edit
        self.on_delete = on_delete
        self.on_select = on_select
        self.selected = selected
        self.compact = compact

    def build(self) -> None:
        """Build the entity card UI."""
        color = get_entity_color(self.entity.type)
        icon = get_entity_icon(self.entity.type)

        # Check for quality scores in attributes
        quality_scores = self.entity.attributes.get("quality_scores")
        avg_quality = quality_scores.get("average") if quality_scores else None

        # Card container
        card_classes = "w-full cursor-pointer hover:shadow-lg transition-shadow"
        if self.selected:
            card_classes += " ring-2 ring-blue-500"

        with ui.card().classes(card_classes).on("click", self._handle_select):
            # Header with type color
            with (
                ui.row()
                .classes("w-full items-center gap-2")
                .style(
                    f"background-color: {color}22; padding: 8px 12px; border-radius: 4px 4px 0 0;"
                )
            ):
                ui.icon(icon).style(f"color: {color};")
                ui.label(self.entity.type.title()).classes("text-sm font-medium").style(
                    f"color: {color};"
                )

                # Quality score badge
                if avg_quality is not None and quality_scores is not None:
                    badge_color = _get_quality_badge_color(avg_quality)
                    tooltip_text = _format_quality_tooltip(quality_scores)
                    with ui.element("div").classes("ml-2"):
                        ui.badge(
                            f"{avg_quality:.1f}",
                            color=badge_color,
                        ).props("outline").tooltip(tooltip_text).classes("text-xs")

                ui.space()

                # Actions
                if self.on_edit:
                    ui.button(
                        icon="edit",
                        on_click=lambda: self._handle_edit(),
                    ).props("flat round size=sm")

                if self.on_delete:
                    ui.button(
                        icon="delete",
                        on_click=lambda: self._handle_delete(),
                    ).props("flat round size=sm color=negative")

            # Body
            with ui.column().classes("p-3 gap-2"):
                # Name
                ui.label(self.entity.name).classes("text-lg font-semibold")

                # Description (truncated in compact mode)
                if self.entity.description:
                    desc = self.entity.description
                    if self.compact and len(desc) > 100:
                        desc = desc[:100] + "..."
                    ui.label(desc).classes("text-sm text-gray-400")

                # Attributes (only in non-compact mode)
                if not self.compact and self.entity.attributes:
                    self._build_attributes()

    def _build_attributes(self) -> None:
        """Build the attributes display."""
        # Filter out quality_scores as it's displayed in the badge
        display_attrs = {k: v for k, v in self.entity.attributes.items() if k != "quality_scores"}

        if not display_attrs:
            return

        with ui.expansion("Attributes", icon="list").classes("w-full"):
            for key, value in display_attrs.items():
                with ui.row().classes("w-full items-start gap-2"):
                    ui.label(f"{key}:").classes("text-sm font-medium text-gray-300")
                    if isinstance(value, list):
                        ui.label(", ".join(str(v) for v in value)).classes("text-sm text-gray-400")
                    else:
                        ui.label(str(value)).classes("text-sm text-gray-400")

    async def _handle_select(self) -> None:
        """Handle card selection."""
        if self.on_select:
            await self.on_select(self.entity)

    async def _handle_edit(self) -> None:
        """Handle edit button click."""
        if self.on_edit:
            await self.on_edit(self.entity)

    async def _handle_delete(self) -> None:
        """Handle delete button click."""
        if self.on_delete:
            await self.on_delete(self.entity)


def entity_list_item(
    entity: Entity,
    on_select: Callable[[Entity], Any] | None = None,
    selected: bool = False,
) -> None:
    """Create a simple list item for an entity.

    Args:
        entity: Entity to display.
        on_select: Callback when item is clicked.
        selected: Whether this item is selected.
    """
    color = get_entity_color(entity.type)
    icon = get_entity_icon(entity.type)

    # Get role-based border styling using centralized helper
    border_style = get_role_border_style(entity.attributes)

    item_classes = "w-full items-center gap-2 p-2 rounded cursor-pointer hover:bg-gray-700"
    if selected:
        item_classes += " bg-blue-900"

    with (
        ui.row()
        .classes(item_classes)
        .style(border_style)
        .on("click", lambda: on_select(entity) if on_select else None)
    ):
        ui.icon(icon, size="sm").style(f"color: {color};")
        ui.label(entity.name).classes("flex-grow truncate")
        ui.badge(entity.type).style(f"background-color: {color}; color: white;")


def entity_badge(entity: Entity) -> None:
    """Create a small badge for an entity.

    Args:
        entity: Entity to display.
    """
    color = get_entity_color(entity.type)
    icon = get_entity_icon(entity.type)

    with (
        ui.row()
        .classes("items-center gap-1")
        .style(f"background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px;")
    ):
        ui.icon(icon, size="xs")
        ui.label(entity.name).classes("text-sm")
