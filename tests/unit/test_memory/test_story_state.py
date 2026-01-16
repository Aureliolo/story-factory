"""Tests for memory/story_state.py."""

from memory.story_state import (
    Character,
    PlotPoint,
    StoryBrief,
    StoryState,
)


class TestCharacter:
    """Tests for Character model."""

    def test_get_arc_summary_empty(self):
        """Test get_arc_summary with no progress and no notes."""
        char = Character(name="Alice", role="protagonist", description="Main character")
        assert char.get_arc_summary() == ""

    def test_get_arc_summary_notes_only(self):
        """Test get_arc_summary with notes but no progress."""
        char = Character(
            name="Alice",
            role="protagonist",
            description="Main character",
            arc_notes="Should grow from timid to confident",
        )
        assert char.get_arc_summary() == "Arc: Should grow from timid to confident"

    def test_get_arc_summary_with_progress(self):
        """Test get_arc_summary with arc progress."""
        char = Character(
            name="Alice",
            role="protagonist",
            description="Main character",
            arc_notes="Hero's journey",
            arc_progress={1: "Ordinary world", 2: "Call to adventure", 3: "Refusing the call"},
        )
        summary = char.get_arc_summary()
        assert "Arc plan: Hero's journey" in summary
        assert "Ch1: Ordinary world" in summary
        assert "Ch2: Call to adventure" in summary
        assert "Ch3: Refusing the call" in summary

    def test_get_arc_summary_progress_without_notes(self):
        """Test get_arc_summary with progress but no notes."""
        char = Character(
            name="Alice",
            role="protagonist",
            description="Main character",
            arc_progress={1: "Introduction", 2: "Development"},
        )
        summary = char.get_arc_summary()
        assert "Arc plan:" not in summary
        assert "Ch1: Introduction" in summary
        assert "Ch2: Development" in summary

    def test_update_arc(self):
        """Test update_arc adds progress."""
        char = Character(name="Bob", role="supporting", description="Friend")
        char.update_arc(1, "First appearance")
        char.update_arc(2, "Shows loyalty")
        assert char.arc_progress == {1: "First appearance", 2: "Shows loyalty"}


class TestStoryState:
    """Tests for StoryState model."""

    def test_get_context_summary_with_characters(self):
        """Test get_context_summary includes character info."""
        state = StoryState(
            id="test",
            status="writing",
            brief=StoryBrief(
                premise="A test story",
                genre="Fantasy",
                tone="Epic",
                setting_time="Medieval",
                setting_place="Kingdom",
                target_length="novella",
                language="English",
                content_rating="general",
            ),
            characters=[
                Character(name="Alice", role="protagonist", description="Hero"),
                Character(name="Bob", role="antagonist", description="Villain"),
            ],
        )
        summary = state.get_context_summary()
        assert "CHARACTERS:" in summary
        assert "Alice (protagonist)" in summary
        assert "Bob (antagonist)" in summary

    def test_get_context_summary_with_plot_points(self):
        """Test get_context_summary includes plot point info."""
        state = StoryState(
            id="test",
            status="writing",
            brief=StoryBrief(
                premise="A test story",
                genre="Fantasy",
                tone="Epic",
                setting_time="Medieval",
                setting_place="Kingdom",
                target_length="novella",
                language="English",
                content_rating="general",
            ),
            plot_points=[
                PlotPoint(description="Hero meets mentor", completed=True),
                PlotPoint(description="Hero faces trial", completed=False),
                PlotPoint(description="Final battle", completed=False),
            ],
        )
        summary = state.get_context_summary()
        assert "COMPLETED PLOT POINTS: 1" in summary
        assert "UPCOMING: Hero faces trial" in summary

    def test_get_context_summary_with_established_facts(self):
        """Test get_context_summary includes established facts."""
        state = StoryState(
            id="test",
            status="writing",
            brief=StoryBrief(
                premise="A test story",
                genre="Fantasy",
                tone="Epic",
                setting_time="Medieval",
                setting_place="Kingdom",
                target_length="novella",
                language="English",
                content_rating="general",
            ),
            established_facts=["The hero is from a small village", "Magic requires sacrifice"],
        )
        summary = state.get_context_summary()
        assert "RECENT FACTS:" in summary
        assert "The hero is from a small village" in summary
        assert "Magic requires sacrifice" in summary

    def test_add_established_fact(self):
        """Test add_established_fact adds fact and updates timestamp."""
        state = StoryState(id="test", status="writing")
        original_time = state.updated_at

        state.add_established_fact("The dragon is actually friendly")

        assert "The dragon is actually friendly" in state.established_facts
        assert state.updated_at >= original_time

    def test_get_character_by_name_found(self):
        """Test get_character_by_name returns character when found."""
        state = StoryState(
            id="test",
            status="writing",
            characters=[
                Character(name="Alice", role="protagonist", description="Hero"),
                Character(name="Bob", role="antagonist", description="Villain"),
            ],
        )
        char = state.get_character_by_name("Alice")
        assert char is not None
        assert char.name == "Alice"
        assert char.role == "protagonist"

    def test_get_character_by_name_case_insensitive(self):
        """Test get_character_by_name is case insensitive."""
        state = StoryState(
            id="test",
            status="writing",
            characters=[Character(name="Alice", role="protagonist", description="Hero")],
        )
        char = state.get_character_by_name("ALICE")
        assert char is not None
        assert char.name == "Alice"

    def test_get_character_by_name_not_found(self):
        """Test get_character_by_name returns None when not found."""
        state = StoryState(
            id="test",
            status="writing",
            characters=[Character(name="Alice", role="protagonist", description="Hero")],
        )
        char = state.get_character_by_name("NonExistent")
        assert char is None
