"""Tests for StoryService."""

from services.story_service import StoryService


class TestStoryService:
    """Tests for StoryService."""

    def test_add_review(self, tmp_settings, sample_story_with_chapters):
        """Test adding a review to a story."""
        service = StoryService(tmp_settings)
        state = sample_story_with_chapters

        # Add a review
        service.add_review(
            state,
            review_type="user_feedback",
            content="This chapter needs more action",
            chapter_num=1,
        )

        # Verify review was added
        assert len(state.reviews) == 1
        assert state.reviews[0]["type"] == "user_feedback"
        assert state.reviews[0]["content"] == "This chapter needs more action"
        assert state.reviews[0]["chapter"] == 1

    def test_add_review_without_chapter(self, tmp_settings, sample_story_with_chapters):
        """Test adding a general review without chapter number."""
        service = StoryService(tmp_settings)
        state = sample_story_with_chapters

        # Add a general review
        service.add_review(
            state,
            review_type="user_note",
            content="Overall pacing is good",
            chapter_num=None,
        )

        # Verify review was added
        assert len(state.reviews) == 1
        assert state.reviews[0]["type"] == "user_note"
        assert state.reviews[0]["content"] == "Overall pacing is good"
        assert state.reviews[0]["chapter"] is None

    def test_add_multiple_reviews(self, tmp_settings, sample_story_with_chapters):
        """Test adding multiple reviews."""
        service = StoryService(tmp_settings)
        state = sample_story_with_chapters

        # Add multiple reviews
        service.add_review(state, "user_feedback", "Chapter 1 feedback", 1)
        service.add_review(state, "user_feedback", "Chapter 2 feedback", 2)
        service.add_review(state, "ai_suggestion", "Consider adding more dialogue", 1)

        # Verify all reviews were added
        assert len(state.reviews) == 3
        assert state.reviews[0]["chapter"] == 1
        assert state.reviews[1]["chapter"] == 2
        assert state.reviews[2]["chapter"] == 1

    def test_get_reviews_all(self, tmp_settings, sample_story_with_chapters):
        """Test getting all reviews."""
        service = StoryService(tmp_settings)
        state = sample_story_with_chapters

        # Add reviews
        service.add_review(state, "user_feedback", "Feedback 1", 1)
        service.add_review(state, "user_feedback", "Feedback 2", 2)

        # Get all reviews
        reviews = service.get_reviews(state)

        assert len(reviews) == 2

    def test_get_reviews_by_chapter(self, tmp_settings, sample_story_with_chapters):
        """Test getting reviews for a specific chapter."""
        service = StoryService(tmp_settings)
        state = sample_story_with_chapters

        # Add reviews for different chapters
        service.add_review(state, "user_feedback", "Chapter 1 feedback", 1)
        service.add_review(state, "user_feedback", "Chapter 2 feedback", 2)
        service.add_review(state, "user_note", "Another chapter 1 note", 1)

        # Get reviews for chapter 1
        chapter_1_reviews = service.get_reviews(state, chapter_num=1)

        assert len(chapter_1_reviews) == 2
        assert all(r["chapter"] == 1 for r in chapter_1_reviews)
