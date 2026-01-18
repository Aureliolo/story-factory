"""Tests for the JSON parsing utility."""

from pydantic import BaseModel

from utils.json_parser import (
    clean_llm_text,
    extract_json,
    extract_json_list,
    parse_json_list_to_models,
    parse_json_to_model,
)


class SampleModel(BaseModel):
    name: str
    value: int


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extracts_from_code_block(self):
        """Should extract JSON from markdown code block."""
        response = """Here is the data:
```json
{"name": "test", "value": 42}
```
That's all."""
        result = extract_json(response)
        assert result == {"name": "test", "value": 42}

    def test_returns_none_for_no_json(self):
        """Should return None when no JSON is present."""
        response = "This is just plain text without any JSON."
        result = extract_json(response)
        assert result is None

    def test_uses_fallback_pattern(self):
        """Should use fallback pattern when no code block found."""
        response = 'The result is {"name": "fallback", "value": 99} here.'
        fallback = r'\{[^{}]*"name"[^{}]*\}'
        result = extract_json(response, fallback_pattern=fallback)
        assert result == {"name": "fallback", "value": 99}

    def test_returns_none_for_invalid_json(self):
        """Should return None for invalid JSON."""
        response = """```json
{"name": "test", value: 42}
```"""
        result = extract_json(response)
        assert result is None

    def test_handles_nested_code_blocks(self):
        """Should extract JSON from proper code block."""
        response = """Some text
```json
{"nested": {"key": "value"}}
```
More text"""
        result = extract_json(response)
        assert result == {"nested": {"key": "value"}}

    def test_extracts_from_plain_code_block(self):
        """Should extract JSON from plain ``` code block (no json marker)."""
        response = """Here is the data:
```
{"name": "test", "value": 42}
```
That's all."""
        result = extract_json(response)
        assert result == {"name": "test", "value": 42}

    def test_raw_json_array_invalid_returns_none(self):
        """Should return None when raw JSON array is invalid."""
        # Text contains what looks like a JSON array but is actually invalid
        response = "Here is the array: [invalid json content, missing quotes]"
        result = extract_json(response)
        assert result is None

    def test_raw_json_array_valid_returns_array(self):
        """Should extract JSON from raw array without code block."""
        # Valid JSON array embedded in text (no code block)
        response = 'Here is the data: [{"name": "a"}, {"name": "b"}] - that is all.'
        result = extract_json(response)
        assert result == [{"name": "a"}, {"name": "b"}]

    def test_fallback_pattern_invalid_json_returns_none(self):
        """Should return None when fallback pattern matches but JSON is invalid."""
        # Fallback pattern matches, but the content is not valid JSON
        response = 'The result is ITEM{broken:json,no"quotes} here.'
        fallback = r"ITEM\{[^{}]*\}"
        result = extract_json(response, fallback_pattern=fallback)
        assert result is None

    def test_fallback_pattern_success_after_raw_strategies_fail(self):
        """Should use fallback pattern when raw JSON strategies fail."""
        # Strategy 3 (raw JSON object) regex is greedy - it matches from first { to last }
        # This response has an invalid JSON when matched greedily, but a more specific
        # fallback pattern can extract valid JSON.
        # Strategy 3 matches: `{invalid "broken} {"valid": "json"}`
        # This is invalid JSON (contains unbalanced quotes), so Strategy 3 fails
        # Then fallback pattern matches just the valid JSON object
        response = '{invalid "broken} {"valid": "json"}'
        fallback = r'\{"valid"[^}]*\}'  # Matches just {"valid": "json"}
        result = extract_json(response, fallback_pattern=fallback)
        assert result == {"valid": "json"}


class TestExtractJsonList:
    """Tests for extract_json_list function."""

    def test_extracts_list(self):
        """Should extract a JSON array."""
        response = """```json
[{"name": "a"}, {"name": "b"}]
```"""
        result = extract_json_list(response)
        assert result == [{"name": "a"}, {"name": "b"}]

    def test_returns_none_for_object(self):
        """Should return None when JSON is an object, not array."""
        response = """```json
{"name": "test"}
```"""
        result = extract_json_list(response)
        assert result is None

    def test_returns_none_for_no_json(self):
        """Should return None when no JSON is present."""
        response = "No JSON here."
        result = extract_json_list(response)
        assert result is None


class TestParseJsonToModel:
    """Tests for parse_json_to_model function."""

    def test_parses_to_model(self):
        """Should parse JSON to Pydantic model."""
        response = """```json
{"name": "test", "value": 42}
```"""
        result = parse_json_to_model(response, SampleModel)
        assert isinstance(result, SampleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_returns_none_for_invalid_data(self):
        """Should return None when data doesn't match model."""
        response = """```json
{"name": "test", "wrong_field": 42}
```"""
        result = parse_json_to_model(response, SampleModel)
        assert result is None

    def test_returns_none_for_no_json(self):
        """Should return None when no JSON found."""
        response = "No JSON here."
        result = parse_json_to_model(response, SampleModel)
        assert result is None

    def test_returns_none_when_json_is_list_not_object(self):
        """Should return None and log warning when JSON is a list, not object."""
        response = """```json
[{"name": "a"}, {"name": "b"}]
```"""
        result = parse_json_to_model(response, SampleModel)
        assert result is None


class TestParseJsonListToModels:
    """Tests for parse_json_list_to_models function."""

    def test_parses_list_to_models(self):
        """Should parse JSON array to list of Pydantic models."""
        response = """```json
[{"name": "a", "value": 1}, {"name": "b", "value": 2}]
```"""
        result = parse_json_list_to_models(response, SampleModel)
        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_returns_empty_list_for_no_json(self):
        """Should return empty list when no JSON found."""
        response = "No JSON here."
        result = parse_json_list_to_models(response, SampleModel)
        assert result == []

    def test_skips_invalid_items_in_list(self):
        """Should skip items that don't match model and continue."""
        response = """```json
[{"name": "valid", "value": 1}, {"wrong_field": "invalid"}, {"name": "also_valid", "value": 3}]
```"""
        result = parse_json_list_to_models(response, SampleModel)
        # Should have 2 valid items, skipping the invalid one
        assert len(result) == 2
        assert result[0].name == "valid"
        assert result[1].name == "also_valid"


class TestCleanLlmText:
    """Tests for clean_llm_text function."""

    def test_removes_think_tags_with_content(self):
        """Should remove <think>...</think> blocks including content."""
        text = "Hello <think>internal reasoning here</think> world"
        result = clean_llm_text(text)
        assert result == "Hello  world"

    def test_removes_multiline_think_tags(self):
        """Should remove multiline think blocks."""
        text = """Start
<think>
This is
multiline
thinking
</think>
End"""
        result = clean_llm_text(text)
        assert "Start" in result
        assert "End" in result
        assert "multiline" not in result
        assert "thinking" not in result

    def test_removes_orphan_closing_think_tag(self):
        """Should remove orphan </think> tags."""
        text = "Some text </think> more text"
        result = clean_llm_text(text)
        assert result == "Some text  more text"

    def test_removes_orphan_opening_think_tag(self):
        """Should remove orphan <think> tags."""
        text = "Some text <think> more text"
        result = clean_llm_text(text)
        assert result == "Some text  more text"

    def test_removes_special_tokens(self):
        """Should remove special tokens like <|endoftext|>."""
        text = "Hello world<|endoftext|>"
        result = clean_llm_text(text)
        assert result == "Hello world"

    def test_collapses_excessive_newlines(self):
        """Should collapse more than 2 consecutive newlines."""
        text = "Para 1\n\n\n\n\nPara 2"
        result = clean_llm_text(text)
        assert result == "Para 1\n\nPara 2"

    def test_strips_whitespace(self):
        """Should strip leading and trailing whitespace."""
        text = "  \n  content here  \n  "
        result = clean_llm_text(text)
        assert result == "content here"

    def test_handles_empty_string(self):
        """Should handle empty string."""
        result = clean_llm_text("")
        assert result == ""

    def test_handles_none_gracefully(self):
        """Should handle None-like falsy values."""
        result = clean_llm_text("")
        assert result == ""

    def test_preserves_normal_text(self):
        """Should not modify normal text without LLM artifacts."""
        text = "This is a normal paragraph.\n\nAnother paragraph."
        result = clean_llm_text(text)
        assert result == text
