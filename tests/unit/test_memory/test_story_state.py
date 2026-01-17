"""Tests for memory/story_state.py."""

from memory.story_state import (
    Chapter,
    Character,
    PlotPoint,
    Scene,
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
                        Scene(number=1, title="Opening", goal="Introduce setting"),
                        Scene(number=2, title="Inciting incident", goal="Start conflict"),
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
                            number=1,
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


class TestScene:
    """Tests for Scene model."""

    def test_scene_creation_minimal(self):
        """Test creating a scene with minimal required fields."""
        scene = Scene(number=1, title="Opening scene", goal="Introduce protagonist")
        assert scene.number == 1
        assert scene.title == "Opening scene"
        assert scene.goal == "Introduce protagonist"
        assert scene.pov_character == ""
        assert scene.location == ""
        assert scene.beats == []
        assert scene.content == ""

    def test_scene_creation_full(self):
        """Test creating a scene with all fields populated."""
        scene = Scene(
            number=2,
            title="The confrontation",
            goal="Build tension between rivals",
            pov_character="Alice",
            location="Town square",
            beats=["Alice arrives", "Bob challenges her", "Crowd gathers"],
            content="Alice stepped into the square...",
        )
        assert scene.number == 2
        assert scene.title == "The confrontation"
        assert scene.goal == "Build tension between rivals"
        assert scene.pov_character == "Alice"
        assert scene.location == "Town square"
        assert len(scene.beats) == 3
        assert scene.beats[0] == "Alice arrives"
        assert scene.content.startswith("Alice stepped")

    def test_scene_serialization(self):
        """Test scene can be serialized to dict/JSON."""
        scene = Scene(
            number=1,
            title="Test scene",
            goal="Test serialization",
            pov_character="Hero",
            location="Castle",
            beats=["Event 1", "Event 2"],
            content="Some content",
        )
        data = scene.model_dump(mode="json")
        assert data["number"] == 1
        assert data["title"] == "Test scene"
        assert data["goal"] == "Test serialization"
        assert data["pov_character"] == "Hero"
        assert data["location"] == "Castle"
        assert data["beats"] == ["Event 1", "Event 2"]
        assert data["content"] == "Some content"

    def test_scene_deserialization(self):
        """Test scene can be deserialized from dict/JSON."""
        data = {
            "number": 3,
            "title": "Finale",
            "goal": "Resolve conflict",
            "pov_character": "Villain",
            "location": "Throne room",
            "beats": ["Final showdown"],
            "content": "The end is near...",
        }
        scene = Scene.model_validate(data)
        assert scene.number == 3
        assert scene.title == "Finale"
        assert scene.pov_character == "Villain"


class TestChapter:
    """Tests for Chapter model."""

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
                Scene(number=1, title="Scene 1", goal="Setup"),
                Scene(number=2, title="Scene 2", goal="Development"),
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
                Scene(number=1, title="Opening", goal="Start story", pov_character="Hero"),
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
                    "number": 1,
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
        assert chapter.scenes[0].title == "Scene A"
        assert chapter.scenes[0].location == "Forest"
        assert chapter.scenes[0].beats == ["Beat 1", "Beat 2"]
=======
        assert scene.word_count == 0
        assert scene.status == "pending"

    def test_scene_creation_with_full_data(self):
        """Test creating a scene with all fields."""
        scene = Scene(
            number=2,
            title="The Confrontation",
            goal="Reveal the antagonist's motivation",
            pov_character="Jack Stone",
            location="Abandoned warehouse",
            beats=["Hero enters", "Villain monologues", "Fight begins"],
            content="Jack stepped into the darkness...",
            word_count=150,
            status="drafted",
        )
        assert scene.number == 2
        assert scene.title == "The Confrontation"
        assert scene.goal == "Reveal the antagonist's motivation"
        assert scene.pov_character == "Jack Stone"
        assert scene.location == "Abandoned warehouse"
        assert len(scene.beats) == 3
        assert "Hero enters" in scene.beats
        assert scene.content == "Jack stepped into the darkness..."
        assert scene.word_count == 150
        assert scene.status == "drafted"


class TestChapterWithScenes:
    """Tests for Chapter model with scene support."""

    def test_chapter_without_scenes(self):
        """Test chapter can be created without scenes."""
        chapter = Chapter(number=1, title="The Beginning", outline="Introduction")
        assert chapter.scenes == []

    def test_chapter_with_scenes(self):
        """Test chapter can contain scenes."""
        scene1 = Scene(number=1, title="Opening", goal="Hook the reader")
        scene2 = Scene(number=2, title="Conflict", goal="Introduce conflict")

        chapter = Chapter(
            number=1,
            title="The Beginning",
            outline="Introduction",
            scenes=[scene1, scene2],
        )

        assert len(chapter.scenes) == 2
        assert chapter.scenes[0].title == "Opening"
        assert chapter.scenes[1].title == "Conflict"

    def test_chapter_backward_compatibility(self):
        """Test that chapters without scenes work as before."""
        chapter = Chapter(
            number=1,
            title="Old Chapter",
            outline="Traditional outline",
            content="Full chapter content here",
        )

        assert chapter.content == "Full chapter content here"
        assert chapter.scenes == []
        assert chapter.status == "pending"

    def test_chapter_serialization_with_scenes(self):
        """Test that chapter with scenes can be serialized/deserialized."""
        scene = Scene(
            number=1,
            title="Test Scene",
            goal="Test goal",
            beats=["beat1", "beat2"],
        )

        chapter = Chapter(
            number=1,
            title="Test Chapter",
            outline="Test outline",
            scenes=[scene],
        )

        # Serialize to dict
        chapter_dict = chapter.model_dump()

        # Verify scene data is present
        assert "scenes" in chapter_dict
        assert len(chapter_dict["scenes"]) == 1
        assert chapter_dict["scenes"][0]["title"] == "Test Scene"
        assert chapter_dict["scenes"][0]["goal"] == "Test goal"

        # Deserialize back
        restored_chapter = Chapter(**chapter_dict)
        assert len(restored_chapter.scenes) == 1
        assert restored_chapter.scenes[0].title == "Test Scene"
        assert restored_chapter.scenes[0].beats == ["beat1", "beat2"]
>>>>>>> ab848f4 (Add Scene model and scene-aware Writer agent)
