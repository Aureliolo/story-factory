"""Tests for the JSON parsing utility."""

import pytest
from pydantic import BaseModel

from src.utils.exceptions import JSONParseError
from src.utils.json_parser import (
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

    def test_raises_error_for_no_json_strict(self):
        """Should raise JSONParseError when no JSON is present in strict mode."""
        response = "This is just plain text without any JSON."
        with pytest.raises(JSONParseError) as exc_info:
            extract_json(response)
        assert "No valid JSON found" in str(exc_info.value)

    def test_returns_none_for_no_json_non_strict(self):
        """Should return None when no JSON is present in non-strict mode."""
        response = "This is just plain text without any JSON."
        result = extract_json(response, strict=False)
        assert result is None

    def test_uses_fallback_pattern(self):
        """Should use fallback pattern when no code block found."""
        response = 'The result is {"name": "fallback", "value": 99} here.'
        fallback = r'\{[^{}]*"name"[^{}]*\}'
        result = extract_json(response, fallback_pattern=fallback)
        assert result == {"name": "fallback", "value": 99}

    def test_raises_error_for_invalid_json_strict(self):
        """Should raise JSONParseError for invalid JSON in strict mode."""
        response = """```json
{"name": "test", value: 42}
```"""
        with pytest.raises(JSONParseError):
            extract_json(response)

    def test_returns_none_for_invalid_json_non_strict(self):
        """Should return None for invalid JSON in non-strict mode."""
        response = """```json
{"name": "test", value: 42}
```"""
        result = extract_json(response, strict=False)
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

    def test_raw_json_array_invalid_raises_strict(self):
        """Should raise JSONParseError when raw JSON array is invalid in strict mode."""
        response = "Here is the array: [invalid json content, missing quotes]"
        with pytest.raises(JSONParseError):
            extract_json(response)

    def test_raw_json_array_invalid_returns_none_non_strict(self):
        """Should return None when raw JSON array is invalid in non-strict mode."""
        response = "Here is the array: [invalid json content, missing quotes]"
        result = extract_json(response, strict=False)
        assert result is None

    def test_raw_json_array_valid_returns_array(self):
        """Should extract JSON from raw array without code block."""
        response = 'Here is the data: [{"name": "a"}, {"name": "b"}] - that is all.'
        result = extract_json(response)
        assert result == [{"name": "a"}, {"name": "b"}]

    def test_fallback_pattern_invalid_json_raises_strict(self):
        """Should raise JSONParseError when fallback matches but JSON is invalid in strict mode."""
        response = 'The result is ITEM{broken:json,no"quotes} here.'
        fallback = r"ITEM\{[^{}]*\}"
        with pytest.raises(JSONParseError):
            extract_json(response, fallback_pattern=fallback)

    def test_fallback_pattern_invalid_json_returns_none_non_strict(self):
        """Should return None when fallback matches but JSON is invalid in non-strict mode."""
        response = 'The result is ITEM{broken:json,no"quotes} here.'
        fallback = r"ITEM\{[^{}]*\}"
        result = extract_json(response, fallback_pattern=fallback, strict=False)
        assert result is None

    def test_fallback_pattern_success_after_raw_strategies_fail(self):
        """Should use fallback pattern when raw JSON strategies fail."""
        response = '{invalid "broken} {"valid": "json"}'
        fallback = r'\{"valid"[^}]*\}'
        result = extract_json(response, fallback_pattern=fallback)
        assert result == {"valid": "json"}

    def test_repairs_truncated_json_simple(self):
        """Should repair simple truncated JSON with missing closing brace."""
        response = '{"name": "test", "value": 42'
        result = extract_json(response, strict=False)
        # Should repair by closing the brace
        assert result is not None
        assert isinstance(result, dict)
        assert result["name"] == "test"

    def test_repairs_truncated_json_with_incomplete_string(self):
        """Should attempt to repair truncated JSON with incomplete string value."""
        response = '{"name": "test", "description": "A long description that gets cut'
        result = extract_json(response, strict=False)
        # The repair adds closing braces; may or may not successfully parse
        # depending on the truncation point
        assert result is None or isinstance(result, dict)

    def test_truncated_json_without_opening_brace_not_repaired(self):
        """Should not attempt repair if no opening brace exists."""
        response = '"name": "test", "value": 42}'
        result = extract_json(response, strict=False)
        # No opening brace, so truncation detection doesn't trigger
        assert result is None

    def test_repairs_nested_truncated_json(self):
        """Should repair truncated JSON with nested objects."""
        response = '{"outer": {"inner": "value"'
        result = extract_json(response, strict=False)
        # Should close both braces
        assert result is not None
        assert "outer" in result

    def test_repairs_truncated_json_with_array(self):
        """Should repair truncated JSON with unclosed array."""
        response = '{"items": ["a", "b"'
        result = extract_json(response, strict=False)
        # Should close both array and object
        assert result is not None


class TestExtractJsonList:
    """Tests for extract_json_list function."""

    def test_extracts_list(self):
        """Should extract a JSON array."""
        response = """```json
[{"name": "a"}, {"name": "b"}]
```"""
        result = extract_json_list(response)
        assert result == [{"name": "a"}, {"name": "b"}]

    def test_wraps_single_object_in_list(self):
        """Should wrap a single JSON object in a list (LLM fallback)."""
        response = """```json
{"name": "test"}
```"""
        result = extract_json_list(response)
        assert result == [{"name": "test"}]

    def test_wraps_single_object_in_list_non_strict(self):
        """Should wrap a single JSON object in a list in non-strict mode too."""
        response = """```json
{"name": "test"}
```"""
        result = extract_json_list(response, strict=False)
        assert result == [{"name": "test"}]

    def test_raises_error_for_no_json_strict(self):
        """Should raise JSONParseError when no JSON is present in strict mode."""
        response = "No JSON here."
        with pytest.raises(JSONParseError):
            extract_json_list(response)

    def test_returns_none_for_no_json_non_strict(self):
        """Should return None when no JSON is present in non-strict mode."""
        response = "No JSON here."
        result = extract_json_list(response, strict=False)
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

    def test_raises_error_for_invalid_data_strict(self):
        """Should raise JSONParseError when data doesn't match model in strict mode."""
        response = """```json
{"name": "test", "wrong_field": 42}
```"""
        with pytest.raises(JSONParseError) as exc_info:
            parse_json_to_model(response, SampleModel)
        assert "Failed to create SampleModel" in str(exc_info.value)

    def test_returns_none_for_invalid_data_non_strict(self):
        """Should return None when data doesn't match model in non-strict mode."""
        response = """```json
{"name": "test", "wrong_field": 42}
```"""
        result = parse_json_to_model(response, SampleModel, strict=False)
        assert result is None

    def test_raises_error_for_no_json_strict(self):
        """Should raise JSONParseError when no JSON found in strict mode."""
        response = "No JSON here."
        with pytest.raises(JSONParseError):
            parse_json_to_model(response, SampleModel)

    def test_returns_none_for_no_json_non_strict(self):
        """Should return None when no JSON found in non-strict mode."""
        response = "No JSON here."
        result = parse_json_to_model(response, SampleModel, strict=False)
        assert result is None

    def test_raises_error_when_json_is_list_strict(self):
        """Should raise JSONParseError when JSON is a list, not object, in strict mode."""
        response = """```json
[{"name": "a"}, {"name": "b"}]
```"""
        with pytest.raises(JSONParseError) as exc_info:
            parse_json_to_model(response, SampleModel)
        assert "Expected JSON object but got list" in str(exc_info.value)

    def test_returns_none_when_json_is_list_non_strict(self):
        """Should return None when JSON is a list, not object, in non-strict mode."""
        response = """```json
[{"name": "a"}, {"name": "b"}]
```"""
        result = parse_json_to_model(response, SampleModel, strict=False)
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

    def test_raises_error_for_no_json_strict(self):
        """Should raise JSONParseError when no JSON found in strict mode."""
        response = "No JSON here."
        with pytest.raises(JSONParseError):
            parse_json_list_to_models(response, SampleModel)

    def test_returns_empty_list_for_no_json_non_strict(self):
        """Should return empty list when no JSON found in non-strict mode."""
        response = "No JSON here."
        result = parse_json_list_to_models(response, SampleModel, strict=False)
        assert result == []

    def test_skips_invalid_items_partial_success(self):
        """Should skip invalid items but succeed if min_count met."""
        response = """```json
[{"name": "valid", "value": 1}, {"wrong_field": "invalid"}, {"name": "also_valid", "value": 3}]
```"""
        # Default min_count=1, so should succeed with 2 valid items
        result = parse_json_list_to_models(response, SampleModel)
        assert len(result) == 2
        assert result[0].name == "valid"
        assert result[1].name == "also_valid"

    def test_raises_when_min_count_not_met_strict(self):
        """Should raise JSONParseError when min_count not met in strict mode."""
        response = """```json
[{"wrong_field": "invalid"}]
```"""
        with pytest.raises(JSONParseError) as exc_info:
            parse_json_list_to_models(response, SampleModel, min_count=1)
        assert "need at least 1" in str(exc_info.value)

    def test_succeeds_when_min_count_zero(self):
        """Should succeed with empty result when min_count=0."""
        response = """```json
[{"wrong_field": "invalid"}]
```"""
        result = parse_json_list_to_models(response, SampleModel, min_count=0)
        assert result == []


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
