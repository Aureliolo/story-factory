"""Private entity delegate methods for WorldQualityService.

Thin wrappers that route entity-specific create/judge/refine calls to the
corresponding module-level functions. Extracted from __init__.py to keep
the main service file under the 1000-line pre-commit limit.
"""

from . import (
    _calendar,
    _chapter_quality,
    _character,
    _concept,
    _event,
    _faction,
    _item,
    _location,
    _plot,
    _relationship,
)


class EntityDelegatesMixin:
    """Mixin providing private entity create/judge/refine delegates.

    Each method is a thin wrapper that passes ``self`` (the service instance)
    to the matching module-level function.
    """

    # -- Private: Character helpers --
    def _create_character(self, story_state, existing_names, temperature, custom_instructions=None):
        """Create a character via LLM at the given temperature."""
        return _character._create_character(
            self, story_state, existing_names, temperature, custom_instructions
        )

    def _judge_character_quality(self, character, story_state, temperature):
        """Judge character quality and return scores."""
        return _character._judge_character_quality(self, character, story_state, temperature)

    def _refine_character(self, character, scores, story_state, temperature):
        """Refine a character based on quality scores."""
        return _character._refine_character(self, character, scores, story_state, temperature)

    # -- Private: Location helpers --
    def _create_location(self, story_state, existing_names, temperature):
        """Create a location via LLM at the given temperature."""
        return _location._create_location(self, story_state, existing_names, temperature)

    def _judge_location_quality(self, location, story_state, temperature):
        """Judge location quality and return scores."""
        return _location._judge_location_quality(self, location, story_state, temperature)

    def _refine_location(self, location, scores, story_state, temperature):
        """Refine a location based on quality scores."""
        return _location._refine_location(self, location, scores, story_state, temperature)

    # -- Private: Faction helpers --
    def _create_faction(self, story_state, existing_names, temperature, existing_locations=None):
        """Create a faction via LLM at the given temperature."""
        return _faction._create_faction(
            self, story_state, existing_names, temperature, existing_locations
        )

    def _judge_faction_quality(self, faction, story_state, temperature):
        """Judge faction quality and return scores."""
        return _faction._judge_faction_quality(self, faction, story_state, temperature)

    def _refine_faction(self, faction, scores, story_state, temperature):
        """Refine a faction based on quality scores."""
        return _faction._refine_faction(self, faction, scores, story_state, temperature)

    # -- Private: Item helpers --
    def _create_item(self, story_state, existing_names, temperature):
        """Create an item via LLM at the given temperature."""
        return _item._create_item(self, story_state, existing_names, temperature)

    def _judge_item_quality(self, item, story_state, temperature):
        """Judge item quality and return scores."""
        return _item._judge_item_quality(self, item, story_state, temperature)

    def _refine_item(self, item, scores, story_state, temperature):
        """Refine an item based on quality scores."""
        return _item._refine_item(self, item, scores, story_state, temperature)

    # -- Private: Concept helpers --
    def _create_concept(self, story_state, existing_names, temperature):
        """Create a concept via LLM at the given temperature."""
        return _concept._create_concept(self, story_state, existing_names, temperature)

    def _judge_concept_quality(self, concept, story_state, temperature):
        """Judge concept quality and return scores."""
        return _concept._judge_concept_quality(self, concept, story_state, temperature)

    def _refine_concept(self, concept, scores, story_state, temperature):
        """Refine a concept based on quality scores."""
        return _concept._refine_concept(self, concept, scores, story_state, temperature)

    # -- Private: Event helpers --
    def _create_event(self, story_state, existing_descriptions, entity_context, temperature):
        """Create an event via LLM at the given temperature."""
        return _event._create_event(
            self, story_state, existing_descriptions, entity_context, temperature
        )

    def _judge_event_quality(self, event, story_state, temperature):
        """Judge event quality and return scores."""
        return _event._judge_event_quality(self, event, story_state, temperature)

    def _refine_event(self, event, scores, story_state, temperature):
        """Refine an event based on quality scores."""
        return _event._refine_event(self, event, scores, story_state, temperature)

    # -- Private: Calendar helpers --
    def _create_calendar(self, story_state, temperature):
        """Create a calendar via LLM at the given temperature."""
        return _calendar._create_calendar(self, story_state, temperature)

    def _judge_calendar_quality(self, calendar, story_state, temperature):
        """Judge calendar quality and return scores."""
        return _calendar._judge_calendar_quality(self, calendar, story_state, temperature)

    def _refine_calendar(self, calendar, scores, story_state, temperature):
        """Refine a calendar based on quality scores."""
        return _calendar._refine_calendar(self, calendar, scores, story_state, temperature)

    # -- Private: Relationship helpers --
    @staticmethod
    def _is_duplicate_relationship(source_name, target_name, existing_rels):
        """Check whether a relationship between source and target already exists."""
        return _relationship._is_duplicate_relationship(source_name, target_name, existing_rels)

    def _create_relationship(
        self, story_state, entity_names, existing_rels, temperature, required_entity=None
    ):
        """Create a relationship via LLM at the given temperature."""
        return _relationship._create_relationship(
            self, story_state, entity_names, existing_rels, temperature, required_entity
        )

    def _judge_relationship_quality(self, relationship, story_state, temperature):
        """Judge relationship quality and return scores."""
        return _relationship._judge_relationship_quality(
            self, relationship, story_state, temperature
        )

    def _refine_relationship(self, relationship, scores, story_state, temperature):
        """Refine a relationship based on quality scores."""
        return _relationship._refine_relationship(
            self, relationship, scores, story_state, temperature
        )

    # -- Private: Plot helpers --
    def _judge_plot_quality(self, plot_outline, story_state, temperature):
        """Judge plot outline quality and return scores."""
        return _plot._judge_plot_quality(self, plot_outline, story_state, temperature)

    def _refine_plot(self, plot_outline, scores, story_state, temperature):
        """Refine a plot outline based on quality scores."""
        return _plot._refine_plot(self, plot_outline, scores, story_state, temperature)

    # -- Private: Chapter helpers --
    def _judge_chapter_quality(self, chapter, story_state, temperature):
        """Judge chapter quality and return scores."""
        return _chapter_quality._judge_chapter_quality(self, chapter, story_state, temperature)

    def _refine_chapter_outline(self, chapter, scores, story_state, temperature):
        """Refine a chapter outline based on quality scores."""
        return _chapter_quality._refine_chapter_outline(
            self, chapter, scores, story_state, temperature
        )
