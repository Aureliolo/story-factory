"""Tests for memory/story_state.py."""

import pytest
from pydantic import ValidationError

from src.memory.story_state import (
    Chapter,
    ChapterList,
    Character,
    CharacterCreation,
    CharacterCreationList,
    CharacterList,
    OutlineVariation,
    PlotPoint,
    PlotPointList,
    Scene,
    StoryBrief,
    StoryState,
)
from src.memory.templates import CharacterTemplate, PersonalityTrait


class TestScene:
    """Tests for Scene model."""

    def test_scene_creation(self):
        """Test creating a scene with default values."""
        scene = Scene(id="scene-1", title="Opening Scene", outline="Hero wakes up")
        assert scene.id == "scene-1"
        assert scene.title == "Opening Scene"
        assert scene.outline == "Hero wakes up"
        assert scene.content == ""
        assert scene.word_count == 0
        assert scene.pov_character == ""
        assert scene.location == ""
        assert scene.goals == []
        assert scene.order == 0
        assert scene.status == "pending"

    def test_scene_with_metadata(self):
        """Test creating a scene with metadata."""
        scene = Scene(
            id="scene-1",
            title="The Confrontation",
            outline="Hero faces villain",
            pov_character="Alice",
            location="Dark Tower",
            goals=["Reveal villain's motivation", "Build tension"],
            order=2,
        )
        assert scene.pov_character == "Alice"
        assert scene.location == "Dark Tower"
        assert len(scene.goals) == 2
        assert "Reveal villain's motivation" in scene.goals
        assert scene.order == 2

    def test_update_word_count(self):
        """Test word count update from content."""
        scene = Scene(
            id="scene-1",
            title="Test",
            outline="Test",
            content="This is a test scene with eight words.",
        )
        scene.update_word_count()
        assert scene.word_count == 8

    def test_update_word_count_empty_content(self):
        """Test word count update with empty content."""
        scene = Scene(id="scene-1", title="Test", outline="Test", content="")
        scene.update_word_count()
        assert scene.word_count == 0


class TestChapter:
    """Tests for Chapter model with scenes."""

    def test_chapter_add_scene(self):
        """Test adding a scene to a chapter."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")
        scene = Scene(id="scene-1", title="Scene 1", outline="Opening")

        chapter.add_scene(scene)

        assert len(chapter.scenes) == 1
        assert chapter.scenes[0].id == "scene-1"
        assert chapter.scenes[0].order == 0

    def test_chapter_add_multiple_scenes(self):
        """Test adding multiple scenes maintains order."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")

        scene1 = Scene(id="scene-1", title="Scene 1", outline="Opening")
        scene2 = Scene(id="scene-2", title="Scene 2", outline="Development")
        scene3 = Scene(id="scene-3", title="Scene 3", outline="Conclusion")

        chapter.add_scene(scene1)
        chapter.add_scene(scene2)
        chapter.add_scene(scene3)

        assert len(chapter.scenes) == 3
        assert chapter.scenes[0].order == 0
        assert chapter.scenes[1].order == 1
        assert chapter.scenes[2].order == 2

    def test_chapter_remove_scene(self):
        """Test removing a scene from a chapter."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")

        scene1 = Scene(id="scene-1", title="Scene 1", outline="Opening")
        scene2 = Scene(id="scene-2", title="Scene 2", outline="Development")
        scene3 = Scene(id="scene-3", title="Scene 3", outline="Conclusion")

        chapter.add_scene(scene1)
        chapter.add_scene(scene2)
        chapter.add_scene(scene3)

        # Remove middle scene
        result = chapter.remove_scene("scene-2")

        assert result is True
        assert len(chapter.scenes) == 2
        assert chapter.scenes[0].id == "scene-1"
        assert chapter.scenes[1].id == "scene-3"
        # Check reordering
        assert chapter.scenes[0].order == 0
        assert chapter.scenes[1].order == 1

    def test_chapter_remove_scene_not_found(self):
        """Test removing a non-existent scene returns False."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")
        scene = Scene(id="scene-1", title="Scene 1", outline="Opening")
        chapter.add_scene(scene)

        result = chapter.remove_scene("nonexistent-scene")

        assert result is False
        assert len(chapter.scenes) == 1

    def test_chapter_reorder_scenes(self):
        """Test reordering scenes."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")

        scene1 = Scene(id="scene-1", title="Scene 1", outline="Opening")
        scene2 = Scene(id="scene-2", title="Scene 2", outline="Development")
        scene3 = Scene(id="scene-3", title="Scene 3", outline="Conclusion")

        chapter.add_scene(scene1)
        chapter.add_scene(scene2)
        chapter.add_scene(scene3)

        # Reorder: 3, 1, 2
        chapter.reorder_scenes(["scene-3", "scene-1", "scene-2"])

        assert chapter.scenes[0].id == "scene-3"
        assert chapter.scenes[0].order == 0
        assert chapter.scenes[1].id == "scene-1"
        assert chapter.scenes[1].order == 1
        assert chapter.scenes[2].id == "scene-2"
        assert chapter.scenes[2].order == 2

    def test_chapter_reorder_scenes_partial(self):
        """Test reordering with missing scene IDs."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")

        scene1 = Scene(id="scene-1", title="Scene 1", outline="Opening")
        scene2 = Scene(id="scene-2", title="Scene 2", outline="Development")
        scene3 = Scene(id="scene-3", title="Scene 3", outline="Conclusion")

        chapter.add_scene(scene1)
        chapter.add_scene(scene2)
        chapter.add_scene(scene3)

        # Reorder with only 2 scenes (scene-2 is missing)
        chapter.reorder_scenes(["scene-3", "scene-1"])

        # Only the specified scenes should remain
        assert len(chapter.scenes) == 2
        assert chapter.scenes[0].id == "scene-3"
        assert chapter.scenes[1].id == "scene-1"

    def test_chapter_get_scene_by_id(self):
        """Test getting a scene by ID."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")
        scene = Scene(id="scene-1", title="Scene 1", outline="Opening")
        chapter.add_scene(scene)

        result = chapter.get_scene_by_id("scene-1")

        assert result is not None
        assert result.id == "scene-1"
        assert result.title == "Scene 1"

    def test_chapter_get_scene_by_id_not_found(self):
        """Test getting a non-existent scene by ID."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")
        scene = Scene(id="scene-1", title="Scene 1", outline="Opening")
        chapter.add_scene(scene)

        result = chapter.get_scene_by_id("nonexistent")

        assert result is None

    def test_chapter_update_word_count_from_scenes(self):
        """Test updating chapter word count from scenes."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")

        scene1 = Scene(
            id="scene-1",
            title="Scene 1",
            outline="Opening",
            content="This is scene one with some words.",
        )
        scene1.update_word_count()

        scene2 = Scene(
            id="scene-2",
            title="Scene 2",
            outline="Development",
            content="This is scene two with more words here.",
        )
        scene2.update_word_count()

        chapter.add_scene(scene1)
        chapter.add_scene(scene2)
        chapter.update_chapter_word_count()

        expected_count = scene1.word_count + scene2.word_count
        assert chapter.word_count == expected_count

    def test_chapter_update_word_count_from_content(self):
        """Test updating chapter word count from direct content when no scenes."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="The beginning",
            content="This is a chapter with direct content not in scenes.",
        )

        chapter.update_chapter_word_count()

        assert chapter.word_count == 10

    def test_chapter_update_word_count_empty(self):
        """Test updating chapter word count when empty."""
        chapter = Chapter(number=1, title="Chapter 1", outline="The beginning")

        chapter.update_chapter_word_count()

        assert chapter.word_count == 0


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

    def test_arc_progress_cleans_invalid_string_keys(self):
        """Test arc_progress validator cleans invalid string keys from LLM responses."""
        # LLMs sometimes return string keys like {"Embracing Power": "..."}
        # instead of integer chapter numbers {1: "..."}
        # Simulate LLM JSON response with mixed key types
        raw_data = {
            "name": "Test",
            "role": "protagonist",
            "description": "Test character",
            "arc_progress": {
                "Embracing Power": "Becomes stronger",
                "1": "Valid chapter 1",  # String "1" should be converted to int 1
                "2": "Valid chapter 2",  # String "2" should be converted to int 2
                "invalid": "Should be skipped",
            },
        }
        char = Character.model_validate(raw_data)
        # Only valid integer keys should remain
        assert char.arc_progress == {1: "Valid chapter 1", 2: "Valid chapter 2"}

    def test_arc_progress_handles_empty(self):
        """Test arc_progress validator handles empty dict."""
        char = Character(
            name="Test",
            role="protagonist",
            description="Test character",
            arc_progress={},
        )
        assert char.arc_progress == {}

    def test_arc_progress_handles_non_dict(self):
        """Test arc_progress validator handles non-dict input (e.g., None or list)."""
        # When LLM returns a non-dict value like a list or string
        raw_data = {
            "name": "Test",
            "role": "protagonist",
            "description": "Test character",
            "arc_progress": ["invalid", "list", "format"],
        }
        char = Character.model_validate(raw_data)
        # Should convert to empty dict
        assert char.arc_progress == {}


class TestPersonalityTrait:
    """Tests for PersonalityTrait model."""

    def test_basic_creation(self):
        """Test creating a PersonalityTrait with explicit category."""
        trait = PersonalityTrait(trait="brave", category="core")
        assert trait.trait == "brave"
        assert trait.category == "core"

    def test_default_category_is_core(self):
        """Test that category defaults to 'core'."""
        trait = PersonalityTrait(trait="clever")
        assert trait.category == "core"

    def test_flaw_category(self):
        """Test creating a flaw trait."""
        trait = PersonalityTrait(trait="arrogant", category="flaw")
        assert trait.category == "flaw"

    def test_quirk_category(self):
        """Test creating a quirk trait."""
        trait = PersonalityTrait(trait="talks to animals", category="quirk")
        assert trait.category == "quirk"

    def test_invalid_category_rejected(self):
        """Test that an invalid category raises ValidationError."""
        with pytest.raises(ValidationError):
            PersonalityTrait(trait="mysterious", category="unknown")


class TestCharacterTemplatePersonalityTraits:
    """Tests for CharacterTemplate personality trait normalization."""

    def test_normalize_passthrough_for_non_list(self):
        """Test that non-list input is passed through for Pydantic to handle."""
        with pytest.raises(ValidationError):
            CharacterTemplate(
                name="Hero",
                role="protagonist",
                description="A brave hero",
                personality_traits="not a list",
            )


class TestCharacterPersonalityTraits:
    """Tests for Character personality trait normalization and access."""

    def test_backward_compat_plain_strings(self):
        """Test that plain string traits are auto-converted to PersonalityTrait objects."""
        char = Character(
            name="Alice",
            role="protagonist",
            description="Hero",
            personality_traits=["brave", "clever", "stubborn"],
        )
        assert len(char.personality_traits) == 3
        assert all(isinstance(t, PersonalityTrait) for t in char.personality_traits)
        assert char.personality_traits[0].trait == "brave"
        assert char.personality_traits[0].category == "core"

    def test_structured_traits(self):
        """Test creating a character with structured PersonalityTrait objects."""
        char = Character(
            name="Bob",
            role="antagonist",
            description="Villain",
            personality_traits=[
                {"trait": "ruthless", "category": "core"},
                {"trait": "overconfident", "category": "flaw"},
                {"trait": "collects butterflies", "category": "quirk"},
            ],
        )
        assert len(char.personality_traits) == 3
        assert char.personality_traits[0].trait == "ruthless"
        assert char.personality_traits[1].category == "flaw"
        assert char.personality_traits[2].category == "quirk"

    def test_trait_names_property(self):
        """Test trait_names returns flat list of names."""
        char = Character(
            name="Alice",
            role="protagonist",
            description="Hero",
            personality_traits=[
                {"trait": "brave", "category": "core"},
                {"trait": "arrogant", "category": "flaw"},
            ],
        )
        assert char.trait_names == ["brave", "arrogant"]

    def test_trait_names_empty(self):
        """Test trait_names returns empty list when no traits."""
        char = Character(name="Alice", role="protagonist", description="Hero")
        assert char.trait_names == []

    def test_traits_by_category(self):
        """Test filtering traits by category."""
        char = Character(
            name="Alice",
            role="protagonist",
            description="Hero",
            personality_traits=[
                {"trait": "brave", "category": "core"},
                {"trait": "loyal", "category": "core"},
                {"trait": "impulsive", "category": "flaw"},
                {"trait": "hums when nervous", "category": "quirk"},
            ],
        )
        assert char.traits_by_category("core") == ["brave", "loyal"]
        assert char.traits_by_category("flaw") == ["impulsive"]
        assert char.traits_by_category("quirk") == ["hums when nervous"]

    def test_traits_by_category_empty(self):
        """Test filtering by category with no matches."""
        char = Character(
            name="Alice",
            role="protagonist",
            description="Hero",
            personality_traits=[{"trait": "brave", "category": "core"}],
        )
        assert char.traits_by_category("flaw") == []

    def test_normalize_passthrough_for_non_list(self):
        """Test that non-list input is passed through for Pydantic to handle."""
        # The validator should pass through non-list values (Pydantic raises the type error)
        with pytest.raises(ValidationError):
            Character(
                name="Alice",
                role="protagonist",
                description="Hero",
                personality_traits="not a list",
            )

    def test_serialization_roundtrip(self):
        """Test that PersonalityTrait data survives serialize/deserialize."""
        char = Character(
            name="Alice",
            role="protagonist",
            description="Hero",
            personality_traits=[
                {"trait": "brave", "category": "core"},
                {"trait": "reckless", "category": "flaw"},
            ],
        )
        data = char.model_dump(mode="json")
        restored = Character.model_validate(data)
        assert len(restored.personality_traits) == 2
        assert restored.personality_traits[0].trait == "brave"
        assert restored.personality_traits[0].category == "core"
        assert restored.personality_traits[1].trait == "reckless"
        assert restored.personality_traits[1].category == "flaw"


class TestCharacterCreation:
    """Tests for CharacterCreation model (no runtime fields)."""

    def test_basic_creation(self):
        """Test creating a CharacterCreation without arc_progress/arc_type."""
        cc = CharacterCreation(
            name="Alice",
            role="protagonist",
            description="A brave hero",
            personality_traits=["brave", "clever"],
            goals=["save the world"],
            arc_notes="Hero's journey",
        )
        assert cc.name == "Alice"
        assert cc.role == "protagonist"
        assert len(cc.personality_traits) == 2
        assert cc.personality_traits[0].trait == "brave"

    def test_to_character_conversion(self):
        """Test converting CharacterCreation to full Character."""
        cc = CharacterCreation(
            name="Alice",
            role="protagonist",
            description="A brave hero",
            personality_traits=[
                {"trait": "brave", "category": "core"},
                {"trait": "reckless", "category": "flaw"},
            ],
            goals=["save the world"],
            relationships={"Bob": "rival"},
            arc_notes="Hero's journey",
        )
        char = cc.to_character()
        assert isinstance(char, Character)
        assert char.name == "Alice"
        assert char.role == "protagonist"
        assert char.description == "A brave hero"
        assert len(char.personality_traits) == 2
        assert char.personality_traits[0].trait == "brave"
        assert char.personality_traits[1].category == "flaw"
        assert char.goals == ["save the world"]
        assert char.relationships == {"Bob": "rival"}
        assert char.arc_notes == "Hero's journey"
        # Runtime fields should be empty defaults
        assert char.arc_type is None
        assert char.arc_progress == {}

    def test_normalize_passthrough_for_non_list(self):
        """Test that non-list input is passed through for Pydantic to handle."""
        with pytest.raises(ValidationError):
            CharacterCreation(
                name="Alice",
                role="protagonist",
                description="Hero",
                personality_traits="not a list",
            )

    def test_schema_excludes_runtime_fields(self):
        """Test that CharacterCreation schema excludes arc_progress and arc_type."""
        schema = CharacterCreation.model_json_schema()
        props = schema["properties"]
        assert "arc_progress" not in props
        assert "arc_type" not in props
        assert "name" in props
        assert "role" in props
        assert "personality_traits" in props


class TestCharacterCreationList:
    """Tests for CharacterCreationList wrapper model."""

    def test_wraps_single_object(self):
        """Test CharacterCreationList wraps a single object in a list."""
        single_char = {"name": "Hero", "role": "protagonist", "description": "A brave hero"}
        result = CharacterCreationList.model_validate(single_char)
        assert len(result.characters) == 1
        assert result.characters[0].name == "Hero"

    def test_accepts_proper_format(self):
        """Test CharacterCreationList accepts properly formatted input."""
        data = {
            "characters": [
                {"name": "Alice", "role": "protagonist", "description": "Hero"},
                {"name": "Bob", "role": "antagonist", "description": "Villain"},
            ]
        }
        result = CharacterCreationList.model_validate(data)
        assert len(result.characters) == 2

    def test_to_characters(self):
        """Test converting CharacterCreationList to list of Characters."""
        data = {
            "characters": [
                {
                    "name": "Alice",
                    "role": "protagonist",
                    "description": "Hero",
                    "personality_traits": [{"trait": "brave", "category": "core"}],
                },
                {
                    "name": "Bob",
                    "role": "antagonist",
                    "description": "Villain",
                    "personality_traits": [{"trait": "cunning", "category": "flaw"}],
                },
            ]
        }
        result = CharacterCreationList.model_validate(data)
        characters = result.to_characters()
        assert len(characters) == 2
        assert all(isinstance(c, Character) for c in characters)
        assert characters[0].name == "Alice"
        assert characters[0].arc_type is None
        assert characters[0].arc_progress == {}
        assert characters[1].name == "Bob"

    def test_schema_excludes_runtime_fields(self):
        """Test that CharacterCreationList schema excludes runtime fields."""
        schema = CharacterCreationList.model_json_schema()
        # Walk the schema to find CharacterCreation properties
        assert "$defs" in schema, "Expected $defs in schema"
        assert "CharacterCreation" in schema["$defs"], "Expected CharacterCreation in $defs"
        props = schema["$defs"]["CharacterCreation"]["properties"]
        assert "arc_progress" not in props
        assert "arc_type" not in props


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

    def test_story_state_with_chapters_with_scenes(self):
        """Test StoryState can contain chapters with scenes."""
        state = StoryState(
            id="test",
            status="writing",
            chapters=[
                Chapter(
                    number=1,
                    title="Chapter One",
                    outline="First chapter outline",
                    scenes=[
                        Scene(id="scene-1", title="Opening", goal="Introduce setting"),
                        Scene(id="scene-2", title="Inciting incident", goal="Start conflict"),
                    ],
                )
            ],
        )
        assert len(state.chapters) == 1
        assert len(state.chapters[0].scenes) == 2
        assert state.chapters[0].scenes[0].title == "Opening"

    def test_story_state_serialization_with_scenes(self):
        """Test StoryState with scenes can be serialized and deserialized."""
        state = StoryState(
            id="test-project",
            project_name="Test Story",
            status="writing",
            chapters=[
                Chapter(
                    number=1,
                    title="Prologue",
                    outline="Setup",
                    content="Some content",
                    scenes=[
                        Scene(
                            id="scene-dawn",
                            title="Dawn breaks",
                            goal="Set atmosphere",
                            pov_character="Narrator",
                            location="Mountain peak",
                            beats=["Sun rises", "Hero awakens"],
                            content="The sun rose over...",
                        )
                    ],
                )
            ],
        )

        # Serialize
        data = state.model_dump(mode="json")
        assert data["id"] == "test-project"
        assert len(data["chapters"]) == 1
        assert len(data["chapters"][0]["scenes"]) == 1
        assert data["chapters"][0]["scenes"][0]["location"] == "Mountain peak"

        # Deserialize
        restored = StoryState.model_validate(data)
        assert restored.id == "test-project"
        assert len(restored.chapters) == 1
        assert len(restored.chapters[0].scenes) == 1
        assert restored.chapters[0].scenes[0].title == "Dawn breaks"
        assert restored.chapters[0].scenes[0].beats == ["Sun rises", "Hero awakens"]

    def test_backward_compatibility_old_project_without_scenes(self):
        """Test loading old project JSON without scenes field maintains compatibility."""
        # Simulate old project data structure (before scenes were added)
        old_data = {
            "id": "old-project",
            "created_at": "2025-01-01T12:00:00",
            "updated_at": "2025-01-01T12:00:00",
            "project_name": "Old Project",
            "project_description": "",
            "last_saved": None,
            "world_db_path": "",
            "interview_history": [],
            "reviews": [],
            "brief": {
                "premise": "Old story",
                "genre": "Fantasy",
                "subgenres": [],
                "tone": "Dark",
                "themes": [],
                "setting_time": "Medieval",
                "setting_place": "Europe",
                "target_length": "novel",
                "language": "English",
                "content_rating": "mild",
                "content_preferences": [],
                "content_avoid": [],
                "additional_notes": "",
            },
            "world_description": "",
            "world_rules": [],
            "characters": [],
            "plot_summary": "",
            "plot_points": [],
            "chapters": [
                {
                    "number": 1,
                    "title": "First Chapter",
                    "outline": "Outline text",
                    "content": "Chapter content here",
                    "word_count": 1500,
                    "status": "final",
                    "revision_notes": ["Revised for clarity"],
                    # Note: no "scenes" field in old data
                }
            ],
            "current_chapter": 0,
            "timeline": [],
            "established_facts": [],
            "status": "complete",
        }

        # Should load without errors
        state = StoryState.model_validate(old_data)
        assert state.id == "old-project"
        assert len(state.chapters) == 1
        assert state.chapters[0].scenes == []  # Should default to empty list
        assert state.chapters[0].title == "First Chapter"
        assert state.chapters[0].content == "Chapter content here"


class TestSceneSerialization:
    """Tests for Scene model serialization and deserialization."""

    def test_scene_creation_minimal(self):
        """Test creating a scene with minimal required fields."""
        scene = Scene(id="scene-1", title="Opening scene", goal="Introduce protagonist")
        assert scene.id == "scene-1"
        assert scene.title == "Opening scene"
        assert scene.goal == "Introduce protagonist"
        assert scene.pov_character == ""
        assert scene.location == ""
        assert scene.beats == []
        assert scene.content == ""
        assert scene.order == 0

    def test_scene_creation_full(self):
        """Test creating a scene with all fields populated."""
        scene = Scene(
            id="scene-2",
            title="The confrontation",
            goal="Build tension between rivals",
            pov_character="Alice",
            location="Town square",
            beats=["Alice arrives", "Bob challenges her", "Crowd gathers"],
            content="Alice stepped into the square...",
            order=1,
        )
        assert scene.id == "scene-2"
        assert scene.title == "The confrontation"
        assert scene.goal == "Build tension between rivals"
        assert scene.pov_character == "Alice"
        assert scene.location == "Town square"
        assert len(scene.beats) == 3
        assert scene.beats[0] == "Alice arrives"
        assert scene.content.startswith("Alice stepped")
        assert scene.order == 1

    def test_scene_serialization(self):
        """Test scene can be serialized to dict/JSON."""
        scene = Scene(
            id="scene-1",
            title="Test scene",
            goal="Test serialization",
            pov_character="Hero",
            location="Castle",
            beats=["Event 1", "Event 2"],
            content="Some content",
        )
        data = scene.model_dump(mode="json")
        assert data["id"] == "scene-1"
        assert data["title"] == "Test scene"
        assert data["goal"] == "Test serialization"
        assert data["pov_character"] == "Hero"
        assert data["location"] == "Castle"
        assert data["beats"] == ["Event 1", "Event 2"]
        assert data["content"] == "Some content"

    def test_scene_deserialization(self):
        """Test scene can be deserialized from dict/JSON."""
        data = {
            "id": "scene-3",
            "title": "Finale",
            "goal": "Resolve conflict",
            "pov_character": "Villain",
            "location": "Throne room",
            "beats": ["Final showdown"],
            "content": "The end is near...",
        }
        scene = Scene.model_validate(data)
        assert scene.id == "scene-3"
        assert scene.title == "Finale"
        assert scene.pov_character == "Villain"


class TestChapterSerialization:
    """Tests for Chapter model serialization and backward compatibility."""

    def test_chapter_without_scenes(self):
        """Test creating chapter without scenes (backward compatibility)."""
        chapter = Chapter(
            number=1,
            title="First Chapter",
            outline="Introduction to the world",
            content="Once upon a time...",
            word_count=1000,
            status="drafted",
        )
        assert chapter.number == 1
        assert chapter.scenes == []  # Default empty list

    def test_chapter_with_scenes(self):
        """Test creating chapter with scenes."""
        chapter = Chapter(
            number=1,
            title="First Chapter",
            outline="Introduction",
            scenes=[
                Scene(id="scene-1", title="Scene 1", goal="Setup"),
                Scene(id="scene-2", title="Scene 2", goal="Development"),
            ],
        )
        assert len(chapter.scenes) == 2
        assert chapter.scenes[0].title == "Scene 1"
        assert chapter.scenes[1].title == "Scene 2"

    def test_chapter_serialization_with_scenes(self):
        """Test chapter with scenes can be serialized."""
        chapter = Chapter(
            number=1,
            title="Test Chapter",
            outline="Outline",
            scenes=[
                Scene(id="scene-1", title="Opening", goal="Start story", pov_character="Hero"),
            ],
        )
        data = chapter.model_dump(mode="json")
        assert data["number"] == 1
        assert len(data["scenes"]) == 1
        assert data["scenes"][0]["title"] == "Opening"
        assert data["scenes"][0]["pov_character"] == "Hero"

    def test_chapter_deserialization_without_scenes(self):
        """Test backward compatibility: old chapters without scenes field."""
        # Simulate loading an old chapter JSON that doesn't have scenes field
        data = {
            "number": 1,
            "title": "Old Chapter",
            "outline": "Old outline",
            "content": "Old content",
            "word_count": 500,
            "status": "final",
            "revision_notes": ["Note 1"],
        }
        chapter = Chapter.model_validate(data)
        assert chapter.number == 1
        assert chapter.title == "Old Chapter"
        assert chapter.scenes == []  # Should default to empty list

    def test_chapter_deserialization_with_scenes(self):
        """Test chapter can be deserialized with scenes."""
        data = {
            "number": 2,
            "title": "New Chapter",
            "outline": "Detailed outline",
            "content": "",
            "word_count": 0,
            "status": "pending",
            "revision_notes": [],
            "scenes": [
                {
                    "id": "scene-a",
                    "title": "Scene A",
                    "goal": "Goal A",
                    "pov_character": "Alice",
                    "location": "Forest",
                    "beats": ["Beat 1", "Beat 2"],
                    "content": "Scene content",
                }
            ],
        }
        chapter = Chapter.model_validate(data)
        assert len(chapter.scenes) == 1
        assert chapter.scenes[0].id == "scene-a"
        assert chapter.scenes[0].title == "Scene A"
        assert chapter.scenes[0].location == "Forest"
        assert chapter.scenes[0].beats == ["Beat 1", "Beat 2"]


class TestChapterVersioning:
    """Tests for Chapter versioning functionality."""

    def test_save_current_as_version(self):
        """Test saving current chapter content as a new version."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="Test",
            content="Original content here",
            word_count=3,
        )

        version_id = chapter.save_current_as_version(feedback="Initial save")

        assert version_id is not None
        assert len(chapter.versions) == 1
        assert chapter.current_version_id == version_id
        assert chapter.versions[0].content == "Original content here"
        assert chapter.versions[0].word_count == 3
        assert chapter.versions[0].feedback == "Initial save"
        assert chapter.versions[0].version_number == 1
        assert chapter.versions[0].is_current is True

    def test_save_multiple_versions(self):
        """Test saving multiple versions marks only the latest as current."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="Test",
            content="Version 1 content",
            word_count=3,
        )

        version_1_id = chapter.save_current_as_version()

        # Update content and save another version
        chapter.content = "Version 2 content with more words"
        chapter.word_count = 6
        version_2_id = chapter.save_current_as_version(feedback="Expanded content")

        assert len(chapter.versions) == 2
        assert chapter.current_version_id == version_2_id
        # First version should no longer be current
        assert chapter.versions[0].is_current is False
        assert chapter.versions[0].id == version_1_id
        # Second version should be current
        assert chapter.versions[1].is_current is True
        assert chapter.versions[1].id == version_2_id
        assert chapter.versions[1].version_number == 2

    def test_rollback_to_version_success(self):
        """Test rolling back to a previous version."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="Test",
            content="Original content",
            word_count=2,
        )
        version_1_id = chapter.save_current_as_version()

        # Create a second version
        chapter.content = "New content that we want to undo"
        chapter.word_count = 6
        chapter.save_current_as_version()

        # Rollback to version 1
        result = chapter.rollback_to_version(version_1_id)

        assert result is True
        assert chapter.content == "Original content"
        assert chapter.word_count == 2
        assert chapter.current_version_id == version_1_id
        # Check version states
        assert chapter.versions[0].is_current is True
        assert chapter.versions[1].is_current is False

    def test_rollback_to_version_not_found(self):
        """Test rollback returns False when version not found."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="Test",
            content="Some content",
        )
        chapter.save_current_as_version()

        result = chapter.rollback_to_version("nonexistent-version-id")

        assert result is False

    def test_get_version_by_id_found(self):
        """Test getting a version by ID."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="Test",
            content="Content",
        )
        version_id = chapter.save_current_as_version()

        version = chapter.get_version_by_id(version_id)

        assert version is not None
        assert version.id == version_id
        assert version.content == "Content"

    def test_get_version_by_id_not_found(self):
        """Test getting a nonexistent version returns None."""
        chapter = Chapter(number=1, title="Chapter 1", outline="Test")

        version = chapter.get_version_by_id("nonexistent")

        assert version is None

    def test_get_current_version(self):
        """Test getting the current version."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="Test",
            content="Latest content",
        )
        chapter.save_current_as_version()

        current = chapter.get_current_version()

        assert current is not None
        assert current.content == "Latest content"
        assert current.is_current is True

    def test_get_current_version_none(self):
        """Test getting current version when none exists."""
        chapter = Chapter(number=1, title="Chapter 1", outline="Test")

        current = chapter.get_current_version()

        assert current is None

    def test_compare_versions(self):
        """Test comparing two versions."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="Test",
            content="Short",
            word_count=1,
        )
        version_1_id = chapter.save_current_as_version()

        chapter.content = "A much longer piece of content here"
        chapter.word_count = 7
        version_2_id = chapter.save_current_as_version()

        comparison = chapter.compare_versions(version_1_id, version_2_id)

        assert "version_a" in comparison
        assert "version_b" in comparison
        assert comparison["version_a"]["word_count"] == 1
        assert comparison["version_b"]["word_count"] == 7
        assert comparison["word_count_diff"] == 6

    def test_compare_versions_not_found(self):
        """Test comparing versions when one or both are not found."""
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            outline="Test",
            content="Content",
        )
        version_id = chapter.save_current_as_version()

        # One version missing
        comparison = chapter.compare_versions(version_id, "nonexistent")
        assert "error" in comparison

        # Both versions missing
        comparison = chapter.compare_versions("fake1", "fake2")
        assert "error" in comparison


class TestOutlineVariation:
    """Tests for OutlineVariation model."""

    def test_get_summary_empty(self):
        """Test get_summary with minimal data."""
        variation = OutlineVariation(id="var-1")

        summary = variation.get_summary()

        assert "0 characters, 0 chapters" in summary

    def test_get_summary_with_name(self):
        """Test get_summary includes the name when present."""
        variation = OutlineVariation(id="var-1", name="Dark Ending")

        summary = variation.get_summary()

        assert "**Dark Ending**" in summary

    def test_get_summary_with_short_plot_summary(self):
        """Test get_summary includes full plot summary when short."""
        variation = OutlineVariation(
            id="var-1",
            name="Test",
            plot_summary="A hero saves the world.",
        )

        summary = variation.get_summary()

        assert "A hero saves the world." in summary
        assert "..." not in summary

    def test_get_summary_with_long_plot_summary_truncates(self):
        """Test get_summary truncates long plot summaries."""
        long_plot = "A" * 200  # 200 characters
        variation = OutlineVariation(
            id="var-1",
            name="Test",
            plot_summary=long_plot,
        )

        summary = variation.get_summary()

        # Should truncate to 150 chars + "..."
        assert "A" * 150 + "..." in summary
        assert len(long_plot) > 150  # Confirm it was actually long

    def test_get_summary_with_characters_and_chapters(self):
        """Test get_summary includes character and chapter counts."""
        variation = OutlineVariation(
            id="var-1",
            characters=[
                Character(name="Alice", role="protagonist", description="Hero"),
                Character(name="Bob", role="antagonist", description="Villain"),
            ],
            chapters=[
                Chapter(number=1, title="Ch1", outline="First"),
                Chapter(number=2, title="Ch2", outline="Second"),
                Chapter(number=3, title="Ch3", outline="Third"),
            ],
        )

        summary = variation.get_summary()

        assert "2 characters, 3 chapters" in summary

    def test_get_summary_with_user_rating(self):
        """Test get_summary includes star rating when present."""
        variation = OutlineVariation(
            id="var-1",
            name="Rated Variation",
            user_rating=4,
        )

        summary = variation.get_summary()

        # Should contain 4 stars
        assert "Rating:" in summary

    def test_get_summary_with_zero_rating_excluded(self):
        """Test get_summary excludes rating when zero."""
        variation = OutlineVariation(
            id="var-1",
            name="Unrated",
            user_rating=0,
        )

        summary = variation.get_summary()

        assert "Rating:" not in summary

    def test_get_summary_full(self):
        """Test get_summary with all elements present."""
        variation = OutlineVariation(
            id="var-1",
            name="Epic Fantasy",
            plot_summary="A young wizard discovers their powers and must save the kingdom.",
            characters=[
                Character(name="Wizard", role="protagonist", description="Magic user"),
            ],
            chapters=[
                Chapter(number=1, title="Discovery", outline="Finding powers"),
            ],
            user_rating=5,
        )

        summary = variation.get_summary()

        assert "**Epic Fantasy**" in summary
        assert "A young wizard discovers" in summary
        assert "1 characters, 1 chapters" in summary
        assert "Rating:" in summary


class TestStoryStateVariations:
    """Tests for StoryState outline variation methods."""

    def test_add_outline_variation(self):
        """Test adding an outline variation."""
        state = StoryState(id="test", status="outlining")
        variation = OutlineVariation(
            id="var-1",
            name="First Variation",
            plot_summary="Test plot",
        )
        original_time = state.updated_at

        state.add_outline_variation(variation)

        assert len(state.outline_variations) == 1
        assert state.outline_variations[0].id == "var-1"
        assert state.outline_variations[0].name == "First Variation"
        assert state.updated_at >= original_time

    def test_add_multiple_outline_variations(self):
        """Test adding multiple outline variations."""
        state = StoryState(id="test", status="outlining")
        var1 = OutlineVariation(id="var-1", name="Variation 1")
        var2 = OutlineVariation(id="var-2", name="Variation 2")

        state.add_outline_variation(var1)
        state.add_outline_variation(var2)

        assert len(state.outline_variations) == 2
        assert state.outline_variations[0].id == "var-1"
        assert state.outline_variations[1].id == "var-2"

    def test_get_variation_by_id_found(self):
        """Test getting a variation by ID when it exists."""
        state = StoryState(id="test", status="outlining")
        variation = OutlineVariation(id="var-123", name="Target Variation")
        state.add_outline_variation(variation)

        result = state.get_variation_by_id("var-123")

        assert result is not None
        assert result.id == "var-123"
        assert result.name == "Target Variation"

    def test_get_variation_by_id_not_found(self):
        """Test getting a variation by ID when it doesn't exist."""
        state = StoryState(id="test", status="outlining")
        variation = OutlineVariation(id="var-1", name="Existing")
        state.add_outline_variation(variation)

        result = state.get_variation_by_id("nonexistent-id")

        assert result is None

    def test_get_variation_by_id_empty_list(self):
        """Test getting a variation when no variations exist."""
        state = StoryState(id="test", status="outlining")

        result = state.get_variation_by_id("any-id")

        assert result is None

    def test_select_variation_as_canonical_success(self):
        """Test selecting a variation as canonical copies its data."""
        state = StoryState(id="test", status="outlining")

        # Create a variation with full data
        variation = OutlineVariation(
            id="var-1",
            name="Canonical Choice",
            world_description="A magical realm",
            world_rules=["Magic requires focus", "Dragons are wise"],
            characters=[
                Character(name="Hero", role="protagonist", description="Brave warrior"),
                Character(name="Mentor", role="supporting", description="Old sage"),
            ],
            plot_summary="Hero embarks on a quest",
            plot_points=[
                PlotPoint(description="Hero receives the call"),
                PlotPoint(description="Hero meets mentor"),
            ],
            chapters=[
                Chapter(number=1, title="The Beginning", outline="Introduction"),
                Chapter(number=2, title="The Journey", outline="Adventure starts"),
            ],
        )
        state.add_outline_variation(variation)
        original_time = state.updated_at

        result = state.select_variation_as_canonical("var-1")

        assert result is True
        assert state.selected_variation_id == "var-1"
        assert state.world_description == "A magical realm"
        assert state.world_rules == ["Magic requires focus", "Dragons are wise"]
        assert len(state.characters) == 2
        assert state.characters[0].name == "Hero"
        assert state.characters[1].name == "Mentor"
        assert state.plot_summary == "Hero embarks on a quest"
        assert len(state.plot_points) == 2
        assert len(state.chapters) == 2
        assert state.chapters[0].title == "The Beginning"
        assert state.updated_at >= original_time

    def test_select_variation_as_canonical_not_found(self):
        """Test selecting nonexistent variation returns False."""
        state = StoryState(id="test", status="outlining")
        variation = OutlineVariation(id="var-1", name="Existing")
        state.add_outline_variation(variation)

        result = state.select_variation_as_canonical("nonexistent-id")

        assert result is False
        assert state.selected_variation_id is None

    def test_select_variation_as_canonical_deep_copies_data(self):
        """Test that selecting variation creates deep copies."""
        state = StoryState(id="test", status="outlining")

        character = Character(name="Hero", role="protagonist", description="Original")
        variation = OutlineVariation(
            id="var-1",
            characters=[character],
            chapters=[Chapter(number=1, title="Original Title", outline="Test")],
        )
        state.add_outline_variation(variation)

        state.select_variation_as_canonical("var-1")

        # Modify the original variation's data
        variation.characters[0].description = "Modified"
        variation.chapters[0].title = "Modified Title"

        # State's data should be unchanged (deep copy)
        assert state.characters[0].description == "Original"
        assert state.chapters[0].title == "Original Title"

    def test_create_merged_variation_basic(self):
        """Test creating a merged variation from multiple sources."""
        state = StoryState(id="test", status="outlining")

        # Create source variations
        var1 = OutlineVariation(
            id="var-1",
            name="World Source",
            world_description="Fantasy world",
            world_rules=["Magic exists"],
            characters=[Character(name="Wizard", role="protagonist", description="Powerful")],
        )
        var2 = OutlineVariation(
            id="var-2",
            name="Plot Source",
            plot_summary="Epic quest for the artifact",
            plot_points=[PlotPoint(description="Find the artifact")],
            chapters=[Chapter(number=1, title="Quest Begins", outline="Starting out")],
        )
        state.add_outline_variation(var1)
        state.add_outline_variation(var2)

        merged = state.create_merged_variation(
            name="Best of Both",
            source_variations={
                "var-1": ["world", "characters"],
                "var-2": ["plot", "chapters"],
            },
        )

        assert merged.name == "Best of Both"
        assert merged.world_description == "Fantasy world"
        assert merged.world_rules == ["Magic exists"]
        assert len(merged.characters) == 1
        assert merged.characters[0].name == "Wizard"
        assert merged.plot_summary == "Epic quest for the artifact"
        assert len(merged.plot_points) == 1
        assert len(merged.chapters) == 1
        assert merged.chapters[0].title == "Quest Begins"
        # Should be added to variations list
        assert len(state.outline_variations) == 3
        assert state.outline_variations[2].id == merged.id

    def test_create_merged_variation_with_missing_source(self):
        """Test creating merged variation with a nonexistent source ID."""
        state = StoryState(id="test", status="outlining")

        var1 = OutlineVariation(
            id="var-1",
            name="Existing",
            world_description="Real world",
        )
        state.add_outline_variation(var1)

        merged = state.create_merged_variation(
            name="Partial Merge",
            source_variations={
                "var-1": ["world"],
                "nonexistent": ["characters"],  # This should be skipped
            },
        )

        assert merged.world_description == "Real world"
        assert merged.characters == []  # Not merged from nonexistent

    def test_create_merged_variation_overwrite_warning(self):
        """Test that merging duplicate element types overwrites earlier values."""
        state = StoryState(id="test", status="outlining")

        var1 = OutlineVariation(
            id="var-1",
            name="First",
            world_description="World from var1",
        )
        var2 = OutlineVariation(
            id="var-2",
            name="Second",
            world_description="World from var2",
        )
        state.add_outline_variation(var1)
        state.add_outline_variation(var2)

        # Both sources provide "world" - var2 should overwrite var1
        merged = state.create_merged_variation(
            name="Overwritten",
            source_variations={
                "var-1": ["world"],
                "var-2": ["world"],  # Same element type - will overwrite
            },
        )

        # The second source should have overwritten the first
        assert merged.world_description == "World from var2"

    def test_create_merged_variation_all_element_types(self):
        """Test merging all possible element types."""
        state = StoryState(id="test", status="outlining")

        var1 = OutlineVariation(
            id="var-1",
            world_description="Complete world",
            world_rules=["Rule 1", "Rule 2"],
            characters=[
                Character(name="Alice", role="protagonist", description="Main"),
                Character(name="Bob", role="antagonist", description="Evil"),
            ],
            plot_summary="Complete plot summary",
            plot_points=[
                PlotPoint(description="Point 1"),
                PlotPoint(description="Point 2"),
            ],
            chapters=[
                Chapter(number=1, title="Ch1", outline="First"),
                Chapter(number=2, title="Ch2", outline="Second"),
            ],
        )
        state.add_outline_variation(var1)

        merged = state.create_merged_variation(
            name="Full Merge",
            source_variations={
                "var-1": ["world", "characters", "plot", "chapters"],
            },
        )

        assert merged.world_description == "Complete world"
        assert len(merged.world_rules) == 2
        assert len(merged.characters) == 2
        assert merged.plot_summary == "Complete plot summary"
        assert len(merged.plot_points) == 2
        assert len(merged.chapters) == 2

    def test_create_merged_variation_deep_copies(self):
        """Test that merged variation contains deep copies."""
        state = StoryState(id="test", status="outlining")

        original_char = Character(
            name="Hero",
            role="protagonist",
            description="Original description",
        )
        original_chapter = Chapter(
            number=1,
            title="Original Title",
            outline="Original outline",
        )
        var1 = OutlineVariation(
            id="var-1",
            characters=[original_char],
            chapters=[original_chapter],
            world_rules=["Original rule"],
        )
        state.add_outline_variation(var1)

        merged = state.create_merged_variation(
            name="Deep Copy Test",
            source_variations={"var-1": ["characters", "chapters", "world"]},
        )

        # Modify original variation
        original_char.description = "Modified description"
        original_chapter.title = "Modified Title"
        var1.world_rules.append("New rule")

        # Merged should be unaffected
        assert merged.characters[0].description == "Original description"
        assert merged.chapters[0].title == "Original Title"
        assert len(merged.world_rules) == 1

    def test_create_merged_variation_sets_rationale(self):
        """Test that merged variation has ai_rationale set."""
        state = StoryState(id="test", status="outlining")

        var1 = OutlineVariation(id="var-1")
        var2 = OutlineVariation(id="var-2")
        var3 = OutlineVariation(id="var-3")
        state.add_outline_variation(var1)
        state.add_outline_variation(var2)
        state.add_outline_variation(var3)

        merged = state.create_merged_variation(
            name="Triple Merge",
            source_variations={
                "var-1": ["world"],
                "var-2": ["characters"],
                "var-3": ["plot"],
            },
        )

        assert "Merged from 3 variations" in merged.ai_rationale

    def test_create_merged_variation_empty_sources(self):
        """Test creating merged variation with no valid sources."""
        state = StoryState(id="test", status="outlining")

        merged = state.create_merged_variation(
            name="Empty Merge",
            source_variations={
                "nonexistent-1": ["world"],
                "nonexistent-2": ["characters"],
            },
        )

        # Should still create a variation with default empty values
        assert merged.name == "Empty Merge"
        assert merged.world_description == ""
        assert merged.characters == []
        assert merged.chapters == []


class TestWrapperModelValidators:
    """Tests for list wrapper models that handle single object wrapping."""

    def test_character_list_wraps_single_object(self):
        """Test CharacterList wraps a single Character object in a list."""
        single_char = {"name": "Hero", "role": "protagonist", "description": "A brave hero"}
        result = CharacterList.model_validate(single_char)

        assert len(result.characters) == 1
        assert result.characters[0].name == "Hero"
        assert result.characters[0].role == "protagonist"

    def test_character_list_accepts_proper_format(self):
        """Test CharacterList accepts properly formatted input."""
        proper_data = {
            "characters": [
                {"name": "Alice", "role": "protagonist", "description": "Hero"},
                {"name": "Bob", "role": "antagonist", "description": "Villain"},
            ]
        }
        result = CharacterList.model_validate(proper_data)

        assert len(result.characters) == 2
        assert result.characters[0].name == "Alice"
        assert result.characters[1].name == "Bob"

    def test_plot_point_list_wraps_single_object(self):
        """Test PlotPointList wraps a single PlotPoint object in a list."""
        single_plot = {"description": "Hero discovers the truth"}
        result = PlotPointList.model_validate(single_plot)

        assert len(result.plot_points) == 1
        assert result.plot_points[0].description == "Hero discovers the truth"

    def test_plot_point_list_accepts_proper_format(self):
        """Test PlotPointList accepts properly formatted input."""
        proper_data = {
            "plot_points": [
                {"description": "Plot point 1"},
                {"description": "Plot point 2"},
            ]
        }
        result = PlotPointList.model_validate(proper_data)

        assert len(result.plot_points) == 2

    def test_chapter_list_wraps_single_object(self):
        """Test ChapterList wraps a single Chapter object in a list."""
        single_chapter = {"number": 1, "title": "The Beginning", "outline": "Opening"}
        result = ChapterList.model_validate(single_chapter)

        assert len(result.chapters) == 1
        assert result.chapters[0].number == 1
        assert result.chapters[0].title == "The Beginning"

    def test_chapter_list_accepts_proper_format(self):
        """Test ChapterList accepts properly formatted input."""
        proper_data = {
            "chapters": [
                {"number": 1, "title": "Ch1", "outline": "First"},
                {"number": 2, "title": "Ch2", "outline": "Second"},
            ]
        }
        result = ChapterList.model_validate(proper_data)

        assert len(result.chapters) == 2
        assert result.chapters[0].title == "Ch1"
        assert result.chapters[1].title == "Ch2"
