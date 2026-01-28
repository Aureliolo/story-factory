"""Calendar service - generates fictional calendar systems for story worlds."""

import logging

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.memory.story_state import StoryBrief
from src.memory.world_calendar import (
    CalendarMonth,
    HistoricalEra,
    WorldCalendar,
    create_default_calendar,
)
from src.settings import Settings
from src.utils.exceptions import CalendarGenerationError

logger = logging.getLogger(__name__)


class GeneratedCalendarData(BaseModel):
    """Structured output from calendar generation LLM call."""

    era_name: str = Field(
        description="Name of the current era (e.g., 'Third Age', 'After the Fall')"
    )
    era_abbreviation: str = Field(description="Short abbreviation (e.g., 'TA', 'AF', 'CE')")
    current_year: int = Field(description="Current year in the story timeline")
    months: list[dict] = Field(description="List of months with name, days, and description")
    day_names: list[str] = Field(description="Names for days of the week")
    historical_eras: list[dict] = Field(
        description="List of historical eras with name, start_year, end_year, description"
    )
    date_format: str = Field(
        default="{day} {month}, Year {year} {era}",
        description="Format template for displaying dates",
    )


class CalendarService:
    """Service for generating fictional calendar systems for story worlds.

    Uses LLM to create thematically appropriate calendar systems based on
    the story's genre, setting, and themes.
    """

    def __init__(self, settings: Settings):
        """Initialize calendar service.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._agent: BaseAgent | None = None
        logger.debug("Initialized CalendarService")

    def _get_agent(self) -> BaseAgent:
        """Get or create the calendar generation agent.

        Returns:
            BaseAgent configured for calendar generation.
        """
        if self._agent is None:
            system_prompt = """You are a worldbuilding assistant specialized in creating fictional calendar systems.

Your role is to design immersive, thematically appropriate calendar systems for story worlds. Your calendars should:
- Reflect the culture, technology level, and themes of the setting
- Have evocative month names that fit the world's tone
- Include historical eras that add depth to the timeline
- Feel authentic and lived-in, not generic

Consider:
- Fantasy worlds might have months named after gods, seasons, or celestial events
- Sci-fi settings might use numerical or technical naming conventions
- Historical settings might blend real-world inspiration with fictional elements
- The number of months, days per month, and days per week can vary

Always return your response as valid JSON matching the requested structure."""

            self._agent = BaseAgent(
                name="Calendar Generator",
                role="Worldbuilding Assistant",
                system_prompt=system_prompt,
                agent_role="architect",  # Use architect role for worldbuilding
                settings=self.settings,
            )
            logger.debug("Initialized calendar generation agent")

        return self._agent

    def generate_calendar(
        self,
        story_brief: StoryBrief,
        world_template_name: str | None = None,
    ) -> WorldCalendar:
        """Generate a fictional calendar system based on story context.

        Args:
            story_brief: Story brief with genre, setting, and themes.
            world_template_name: Optional world template name for additional context.

        Returns:
            WorldCalendar with generated content.

        Raises:
            CalendarGenerationError: If calendar generation fails.
        """
        logger.info(f"Generating calendar for {story_brief.genre} story")

        # Build prompt from story context
        context_parts = [
            f"Genre: {story_brief.genre}",
            f"Subgenres: {', '.join(story_brief.subgenres) if story_brief.subgenres else 'None'}",
            f"Premise: {story_brief.premise}",
            f"Tone: {story_brief.tone}",
            f"Setting Time: {story_brief.setting_time}",
            f"Setting Place: {story_brief.setting_place}",
            f"Themes: {', '.join(story_brief.themes) if story_brief.themes else 'None'}",
        ]

        if world_template_name:
            context_parts.append(f"World Template: {world_template_name}")

        context = "\n".join(context_parts)

        prompt = f"""Design a fictional calendar system for this story world.

STORY CONTEXT:
{context}

Create a calendar that:
1. Has 8-14 months with evocative names fitting the setting
2. Includes 5-8 day names for the week
3. Defines 2-5 historical eras with the current era being the most recent
4. Sets an appropriate "current year" for where the story takes place
5. Uses a date format that feels natural for this culture

The calendar should feel authentic to this world - not just a renamed version of our calendar.

For each month, provide:
- name: An evocative name (e.g., "Frostfall", "Highsun", "Reaping Moon")
- days: Number of days (20-40 range works well)
- description: Brief flavor text about this month's nature

For historical eras, provide:
- name: Era name (e.g., "Age of Dragons", "The Long Winter", "Era of Expansion")
- start_year: When this era began
- end_year: When this era ended (null for current era)
- description: What characterized this era

Make the era_abbreviation 2-3 letters that could follow a year number."""

        try:
            agent = self._get_agent()
            result = agent.generate_structured(prompt, GeneratedCalendarData)

            # Convert to WorldCalendar
            months = [
                CalendarMonth(
                    name=m.get("name", f"Month {i + 1}"),
                    days=m.get("days", 30),
                    description=m.get("description", ""),
                )
                for i, m in enumerate(result.months)
            ]

            eras = [
                HistoricalEra(
                    name=e.get("name", f"Era {i + 1}"),
                    start_year=e.get("start_year", 1),
                    end_year=e.get("end_year"),
                    description=e.get("description", ""),
                    display_order=i,
                )
                for i, e in enumerate(result.historical_eras)
            ]

            calendar = WorldCalendar(
                current_era_name=result.era_name,
                era_abbreviation=result.era_abbreviation,
                era_start_year=eras[-1].start_year if eras else 1,
                months=months,
                days_per_week=len(result.day_names),
                day_names=result.day_names,
                current_story_year=result.current_year,
                eras=eras,
                date_format=result.date_format,
            )

            logger.info(
                f"Generated calendar: {result.era_name} ({result.era_abbreviation}), "
                f"{len(months)} months, {len(eras)} eras, current year {result.current_year}"
            )
            return calendar

        except Exception as e:
            logger.exception("Failed to generate calendar")
            raise CalendarGenerationError(f"Failed to generate calendar: {e}") from e

    def generate_calendar_for_genre(self, genre: str) -> WorldCalendar:
        """Generate a basic calendar for a genre without full story context.

        Useful for quick calendar generation when full story brief isn't available.

        Args:
            genre: Story genre (e.g., "Fantasy", "Sci-Fi", "Historical").

        Returns:
            WorldCalendar with genre-appropriate defaults.

        Raises:
            CalendarGenerationError: If calendar generation fails.
        """
        logger.info(f"Generating genre-based calendar for {genre}")

        # Create minimal brief
        brief = StoryBrief(
            premise=f"A {genre.lower()} story",
            genre=genre,
            subgenres=[],
            tone="Epic",
            themes=["Adventure"],
            setting_time="Ancient"
            if genre == "Fantasy"
            else "Future"
            if genre == "Sci-Fi"
            else "Medieval",
            setting_place="Unknown land",
            target_length="novella",
            language="English",
            content_rating="none",
            content_preferences=[],
            content_avoid=[],
        )

        try:
            return self.generate_calendar(brief)
        except CalendarGenerationError:
            # Fall back to default calendar if generation fails
            logger.warning(f"Falling back to default calendar for genre {genre}")
            return create_default_calendar(
                era_name=f"{genre} Era",
                era_abbrev=genre[:2].upper(),
                current_year=1000,
            )
