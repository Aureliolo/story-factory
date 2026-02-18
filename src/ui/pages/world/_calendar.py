"""World calendar settings section.

Provides UI for viewing and managing the world's calendar system.
"""

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from src.memory.world_calendar import WorldCalendar
from src.memory.world_settings import WorldSettings

if TYPE_CHECKING:
    from src.ui.pages.world import WorldPage

logger = logging.getLogger(__name__)


def build_calendar_section(page: WorldPage) -> None:
    """Build the calendar settings section.

    Args:
        page: WorldPage instance.
    """
    logger.debug("Building calendar section")

    with ui.expansion("Calendar & Timeline", icon="calendar_month").classes(
        "w-full"
    ) as calendar_exp:
        page.calendar_expansion = calendar_exp  # type: ignore[attr-defined]

        with ui.column().classes("w-full gap-4 p-4"):
            # Calendar status
            page.calendar_status_container = ui.column().classes("w-full")  # type: ignore[attr-defined]

            # Generate calendar button
            page.generate_calendar_btn = ui.button(  # type: ignore[attr-defined]
                "Generate Calendar",
                icon="auto_awesome",
                on_click=lambda: _generate_calendar(page),
            ).classes("w-full")
            page.generate_calendar_btn.tooltip(  # type: ignore[attr-defined]
                "Generate a fictional calendar system based on your story's genre and setting"
            )

    # Initial refresh
    refresh_calendar_section(page)


def refresh_calendar_section(page: WorldPage) -> None:
    """Refresh the calendar section display.

    Args:
        page: WorldPage instance.
    """
    logger.debug("Refreshing calendar section")

    if not hasattr(page, "calendar_status_container"):
        return

    page.calendar_status_container.clear()

    # Check if we have a world database
    if not page.state.world_db:
        with page.calendar_status_container:
            ui.label("No world loaded. Create or load a project first.").classes("text-gray-500")
        return

    # Try to get calendar from world settings
    calendar = _get_world_calendar(page)

    with page.calendar_status_container:
        if calendar:
            _display_calendar_info(page, calendar)
            page.generate_calendar_btn.text = "Regenerate Calendar"  # type: ignore[attr-defined]
        else:
            ui.label("No calendar configured for this world.").classes("text-gray-500")
            ui.label("Generate a calendar to add timeline context to your entities.").classes(
                "text-sm text-gray-400"
            )
            page.generate_calendar_btn.text = "Generate Calendar"  # type: ignore[attr-defined]


def _get_world_calendar(page: WorldPage) -> WorldCalendar | None:
    """Get the calendar from world settings if available.

    Args:
        page: WorldPage instance.

    Returns:
        WorldCalendar or None.
    """
    logger.debug("Attempting to load world calendar from world settings")

    world_db = page.state.world_db
    if not world_db:
        logger.debug("No world database available")
        return None

    try:
        world_settings = world_db.get_world_settings()
        if world_settings and world_settings.calendar:
            logger.debug(f"Loaded calendar: {world_settings.calendar.current_era_name}")
            return world_settings.calendar
        logger.debug("No calendar in world settings")
        return None
    except Exception as e:
        logger.warning("Could not load world calendar: %s", e, exc_info=True)
        return None


def _display_calendar_info(page: WorldPage, calendar: WorldCalendar) -> None:
    """Display calendar information.

    Args:
        page: WorldPage instance.
        calendar: WorldCalendar instance.
    """
    with ui.card().classes("w-full p-4"):
        ui.label("World Calendar").classes("text-lg font-bold")

        with ui.grid(columns=2).classes("gap-4 mt-2"):
            # Era info
            ui.label("Current Era:").classes("font-medium")
            ui.label(f"{calendar.current_era_name} ({calendar.era_abbreviation})")

            ui.label("Current Year:").classes("font-medium")
            ui.label(f"{calendar.current_story_year} {calendar.era_abbreviation}")

            ui.label("Months:").classes("font-medium")
            ui.label(f"{len(calendar.months)} months ({calendar.total_days_per_year} days/year)")

            ui.label("Week:").classes("font-medium")
            ui.label(f"{calendar.days_per_week} days")

        # Historical eras
        if calendar.eras:
            ui.separator().classes("my-4")
            ui.label("Historical Eras").classes("font-medium")
            with ui.column().classes("gap-2 mt-2"):
                for era in calendar.eras:
                    era_text = f"{era.name}: {era.start_year}"
                    if era.end_year:
                        era_text += f" - {era.end_year}"
                    else:
                        era_text += " - present"
                    ui.label(era_text).classes("text-sm")
                    if era.description:
                        ui.label(era.description).classes("text-xs text-gray-500 ml-4")

        # Month list (collapsible)
        if calendar.months:
            with ui.expansion("View Months", icon="list").classes("w-full mt-4"):
                with ui.column().classes("gap-1"):
                    for i, month in enumerate(calendar.months, 1):
                        ui.label(f"{i}. {month.name} ({month.days} days)").classes("text-sm")
                        if month.description:
                            ui.label(month.description).classes("text-xs text-gray-500 ml-4")


async def _generate_calendar(page: WorldPage) -> None:
    """Generate a calendar for the current world using the quality refinement loop.

    Args:
        page: WorldPage instance.
    """
    from nicegui import run

    logger.info("Generating calendar for world via quality loop")

    # Check if story state has a brief
    story_state = getattr(page.state, "story_state", None)
    if not story_state or not getattr(story_state, "brief", None):
        ui.notify("No story brief available. Complete the interview first.", type="warning")
        return

    if not page.state.world_db:
        ui.notify("No world database loaded.", type="negative")
        return

    # Show loading state
    page.generate_calendar_btn.props("loading")  # type: ignore[attr-defined]
    page.generate_calendar_btn.disable()  # type: ignore[attr-defined]

    page.state.begin_background_task("generate_calendar")
    try:
        # Generate calendar using quality service - run off event loop to avoid blocking
        calendar_dict, scores, iterations = await run.io_bound(
            page.services.world_quality.generate_calendar_with_quality, story_state
        )

        # Convert dict to WorldCalendar and save to world settings
        calendar = WorldCalendar.from_dict(calendar_dict)
        world_settings = page.state.world_db.get_world_settings()
        if world_settings:
            world_settings.calendar = calendar
        else:
            world_settings = WorldSettings(calendar=calendar)
        page.state.world_db.save_world_settings(world_settings)

        # Keep service context in sync so downstream entity generation uses the new calendar
        page.services.world_quality.set_calendar_context(calendar_dict)

        logger.info(
            "Generated and saved calendar: %s (quality: %.1f, iterations: %d)",
            calendar.current_era_name,
            scores.average,
            iterations,
        )

        ui.notify(
            f"Generated calendar: {calendar.current_era_name} "
            f"({calendar.era_abbreviation}) - quality: {scores.average:.1f}/10",
            type="positive",
        )

        # Refresh the display
        refresh_calendar_section(page)

    except Exception as e:
        logger.exception("Failed to generate calendar")
        user_msg = str(e)[:150] if str(e) else "Unknown error"
        ui.notify(f"Failed to generate calendar: {user_msg}", type="negative")

    finally:
        page.state.end_background_task("generate_calendar")
        page.generate_calendar_btn.props(remove="loading")  # type: ignore[attr-defined]
        page.generate_calendar_btn.enable()  # type: ignore[attr-defined]
