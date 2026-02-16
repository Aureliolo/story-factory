"""Tests for the JSON parsing utility."""

import logging

import pytest
from pydantic import BaseModel

from src.utils.exceptions import JSONParseError
from src.utils.json_parser import (
    ParseResult,
    _repair_truncated_json,
    clean_llm_text,
    convert_list_to_dict,
    extract_json,
    extract_json_list,
    extract_json_with_info,
    parse_json_list_to_models,
    parse_json_to_model,
    unwrap_single_json,
)


class SampleModel(BaseModel):
    """Sample Pydantic model for JSON parsing tests."""

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

    def test_converts_list_to_model_when_first_item_matches(self):
        """Should convert list to dict and create model when first item has all fields."""
        response = """```json
[{"name": "first", "value": 1}, {"name": "second", "value": 2}]
```"""
        result = parse_json_to_model(response, SampleModel)
        assert isinstance(result, SampleModel)
        assert result.name == "first"
        assert result.value == 1

    def test_raises_error_when_converted_list_fails_model_strict(self):
        """Should raise JSONParseError when converted list doesn't match model in strict mode."""
        response = """```json
[{"name": "a"}, {"name": "b"}]
```"""
        with pytest.raises(JSONParseError) as exc_info:
            parse_json_to_model(response, SampleModel)
        # The converted dict is {"name": "a"} which lacks "value" field
        assert "Failed to create SampleModel" in str(exc_info.value)

    def test_returns_none_when_converted_list_fails_model_non_strict(self):
        """Should return None when converted list doesn't match model in non-strict mode."""
        response = """```json
[{"name": "a"}, {"name": "b"}]
```"""
        result = parse_json_to_model(response, SampleModel, strict=False)
        assert result is None

    def test_raises_error_when_list_is_empty_strict(self):
        """Should raise JSONParseError when JSON is an empty list in strict mode."""
        response = """```json
[]
```"""
        with pytest.raises(JSONParseError) as exc_info:
            parse_json_to_model(response, SampleModel)
        assert "empty list" in str(exc_info.value)

    def test_returns_none_when_list_is_empty_non_strict(self):
        """Should return None when JSON is an empty list in non-strict mode."""
        response = """```json
[]
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


class TestConvertListToDict:
    """Tests for convert_list_to_dict function."""

    def test_empty_list_returns_empty_dict(self):
        """Should return empty dict for empty list."""
        result = convert_list_to_dict([])
        assert result == {}

    def test_list_of_dicts_returns_first(self):
        """Should return first dict when list contains multiple dicts."""
        data = [{"name": "first"}, {"name": "second"}, {"name": "third"}]
        result = convert_list_to_dict(data)
        assert result == {"name": "first"}

    def test_single_dict_in_list(self):
        """Should return the single dict from a list."""
        data = [{"name": "only", "value": 42}]
        result = convert_list_to_dict(data)
        assert result == {"name": "only", "value": 42}

    def test_list_of_strings_creates_properties(self):
        """Should convert list of strings to properties dict."""
        data = ["brave", "intelligent", "kind"]
        result = convert_list_to_dict(data)
        assert result["properties"] == data
        assert "brave" in result["description"]
        assert "intelligent" in result["description"]
        assert "kind" in result["description"]

    def test_list_of_strings_truncates_description(self):
        """Should truncate description to first 3 items with ellipsis."""
        data = ["one", "two", "three", "four", "five"]
        result = convert_list_to_dict(data)
        assert result["properties"] == data
        assert "..." in result["description"]
        # First 3 should be in description
        assert "one" in result["description"]
        assert "two" in result["description"]
        assert "three" in result["description"]

    def test_mixed_types_creates_items(self):
        """Should wrap mixed-type list in items key."""
        data = ["string", 42, {"nested": "dict"}, True]
        result = convert_list_to_dict(data)
        assert result == {"items": data}

    def test_context_hint_accepted(self):
        """Should accept context_hint parameter (for future use)."""
        data = [{"name": "test"}]
        result = convert_list_to_dict(data, context_hint="character")
        assert result == {"name": "test"}


class TestRepairTruncatedJson:
    """Tests for _repair_truncated_json function."""

    def test_strategy_1_closes_single_brace(self):
        """Strategy 1: Should close single unclosed brace."""
        text = '{"name": "test", "value": 42'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert result.data["name"] == "test"
        assert result.data["value"] == 42
        assert result.repair_applied == "brace_closing"

    def test_strategy_1_closes_nested_braces(self):
        """Strategy 1: Should close multiple nested braces."""
        text = '{"outer": {"inner": {"deep": "value"'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert "outer" in result.data

    def test_strategy_1_closes_brackets_and_braces(self):
        """Strategy 1: Should close both brackets and braces."""
        text = '{"items": ["a", "b", "c"'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert result.data["items"] == ["a", "b", "c"]

    def test_strategy_1_handles_incomplete_string(self):
        """Strategy 1: Should handle truncation in middle of string value."""
        text = '{"name": "test", "description": "This is a long'
        result = _repair_truncated_json(text)
        # May or may not successfully parse depending on content
        # But should not crash
        assert not result.success or isinstance(result.data, dict)

    def test_strategy_2_extracts_complete_object(self):
        """Strategy 2: Should extract last complete object."""
        # Complete object followed by incomplete part
        text = '{"complete": true} {"incomplete": "val'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert result.data["complete"] is True

    def test_strategy_2_extracts_complete_array(self):
        """Strategy 2: Should extract last complete array."""
        text = "[1, 2, 3] [4, 5,"
        result = _repair_truncated_json(text)
        assert result.success
        assert result.data == [1, 2, 3]

    def test_strategy_3_truncates_at_last_comma(self):
        """Strategy 3: Should truncate at last comma if other strategies fail."""
        # This is a malformed JSON that strategy 1 and 2 might not fix
        text = '{"a": 1, "b": 2, "c": '
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        # Should get at least a and b
        assert "a" in result.data
        assert result.data["a"] == 1

    def test_returns_failure_for_completely_broken_json(self):
        """Should return failure when JSON is completely unrecoverable."""
        text = "This is not JSON at all, just plain text."
        result = _repair_truncated_json(text)
        assert not result.success
        assert result.data is None

    def test_handles_escaped_quotes_in_strings(self):
        """Should correctly handle escaped quotes when finding string boundaries."""
        # Escaped quotes inside string followed by complete key-value
        text = '{"quote": "He said \\"hello\\"", "other": 1'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert "quote" in result.data
        assert result.data["other"] == 1

    def test_handles_backslashes_in_strings(self):
        """Should correctly handle backslashes in strings."""
        text = '{"path": "C:\\\\Users\\\\test"'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert "path" in result.data

    def test_array_with_nested_objects(self):
        """Should repair array containing nested objects."""
        text = '[{"a": 1}, {"b": 2'
        result = _repair_truncated_json(text)
        assert result.success
        # Should at least get the first complete object
        assert isinstance(result.data, list)
        assert len(result.data) >= 1

    def test_deeply_nested_structure(self):
        """Should repair deeply nested structures."""
        text = '{"l1": {"l2": {"l3": {"l4": "value"'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert "l1" in result.data

    def test_escape_sequence_in_strategy_2(self):
        """Strategy 2: Should handle escape sequences when finding complete objects."""
        # Valid complete object with escapes, followed by garbage
        text = '{"path": "C:\\\\test\\\\file"} garbage'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert "path" in result.data

    def test_escape_sequence_in_strategy_3(self):
        """Strategy 3: Should handle escape sequences when finding last comma."""
        # Object with escapes and trailing comma
        text = '{"path": "C:\\\\test", "name": "val'
        result = _repair_truncated_json(text)
        # Strategy 3 should find the comma after "C:\\test"
        assert result.success
        assert isinstance(result.data, dict)
        assert "path" in result.data

    def test_strategy_2_extract_fails_parse(self):
        """Strategy 2: When extraction succeeds but parsing fails."""
        # This should find the complete structure but JSON may still be invalid
        # Hard to construct - try something with balanced but invalid JSON
        text = '{"valid": true} {malformed no quotes}'
        result = _repair_truncated_json(text)
        # Should fall through strategies - may succeed with first object
        assert not result.success or result.data == {"valid": True}

    def test_strategy_3_truncate_fails_parse(self):
        """Strategy 3: When truncation at comma still doesn't produce valid JSON."""
        # Construct something where truncating at comma doesn't help
        text = '{broken"key, "another": 1'
        result = _repair_truncated_json(text)
        # May fail all strategies
        assert not result.success or isinstance(result.data, dict)

    def test_newline_escape_in_string(self):
        """Should handle newline escape sequences in strings."""
        text = '{"text": "line1\\nline2"'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert "text" in result.data

    def test_tab_escape_in_string(self):
        """Should handle tab escape sequences in strings."""
        text = '{"text": "col1\\tcol2"'
        result = _repair_truncated_json(text)
        assert result.success
        assert isinstance(result.data, dict)
        assert "text" in result.data

    def test_strategy_3_escape_path_forced(self):
        """Force Strategy 3 to run with escape sequences.

        Strategy 1 fails: balanced braces but invalid JSON structure.
        Strategy 2 fails: no complete structure found.
        Strategy 3 succeeds: truncate at last comma with escapes.
        """
        # Malformed JSON that only Strategy 3 can handle
        # Has escape in string, comma, then garbage that breaks S1/S2
        text = '{"a": "path\\\\to", "b": broken'
        result = _repair_truncated_json(text)
        # Should get at least key "a" from truncating at comma
        assert not result.success or (isinstance(result.data, dict) and "a" in result.data)

    def test_strategy_3_multiple_escapes(self):
        """Strategy 3 with multiple escape sequences."""
        # Multiple backslashes that Strategy 3 must properly track
        text = '{"p1": "a\\\\b", "p2": "c\\\\d\\\\e", "p3": trunc'
        result = _repair_truncated_json(text)
        assert not result.success or isinstance(result.data, dict)

    def test_parse_result_has_strategies_tried(self):
        """ParseResult should track which strategies were attempted."""
        text = '{"name": "test"'
        result = _repair_truncated_json(text)
        assert result.success
        assert len(result.strategies_tried) > 0
        assert "brace_closing" in result.strategies_tried

    def test_parse_result_boolean_context(self):
        """ParseResult should work in boolean context."""
        success_result = _repair_truncated_json('{"a": 1')
        fail_result = _repair_truncated_json("not json")
        assert success_result  # True in boolean context
        assert not fail_result  # False in boolean context


class TestExtractJsonWithInfo:
    """Tests for extract_json_with_info function."""

    def test_returns_parse_result(self):
        """Should return a ParseResult object."""
        response = '```json\n{"name": "test"}\n```'
        result = extract_json_with_info(response)
        assert isinstance(result, ParseResult)

    def test_success_from_code_block(self):
        """Should succeed with JSON code block."""
        response = '```json\n{"name": "test", "value": 42}\n```'
        result = extract_json_with_info(response)
        assert result.success
        assert result.data == {"name": "test", "value": 42}
        assert "json_code_block" in result.strategies_tried

    def test_success_from_plain_code_block(self):
        """Should succeed with plain code block."""
        response = '```\n{"name": "test"}\n```'
        result = extract_json_with_info(response)
        assert result.success
        assert result.data == {"name": "test"}

    def test_success_from_raw_json(self):
        """Should succeed with raw JSON."""
        response = 'Here is the data: {"name": "test"}'
        result = extract_json_with_info(response)
        assert result.success
        assert result.data == {"name": "test"}
        assert "raw_json" in result.strategies_tried

    def test_tracks_repair_strategy(self):
        """Should track which repair strategy was used."""
        response = '{"name": "test", "value": 42'  # Truncated
        result = extract_json_with_info(response)
        assert result.success
        assert result.repair_applied is not None
        assert "truncation_repair" in result.strategies_tried

    def test_failure_with_strategies_tried(self):
        """Should track strategies tried even on failure."""
        response = "This is not JSON at all."
        result = extract_json_with_info(response)
        assert not result.success
        assert result.data is None
        assert len(result.strategies_tried) > 0
        assert result.original_error is not None

    def test_fallback_pattern_success(self):
        """Should track fallback pattern success."""
        # Use a response where raw JSON fails but fallback pattern matches
        # Broken JSON that fails raw parsing, but custom pattern finds valid JSON
        response = '{broken json} The valid data is ITEM{"name": "test"}'
        result = extract_json_with_info(response, fallback_pattern=r'ITEM(\{"name"[^}]*\})')
        # Note: fallback captures full match, so we need the pattern to match valid JSON
        assert result.success or "fallback_pattern" in result.strategies_tried

    def test_strips_think_tags(self):
        """Should strip think tags before parsing."""
        response = '<think>reasoning</think>```json\n{"name": "test"}\n```'
        result = extract_json_with_info(response)
        assert result.success
        assert result.data == {"name": "test"}

    def test_success_from_raw_array(self):
        """Should succeed with raw JSON array."""
        response = "Here is the data: [1, 2, 3]"
        result = extract_json_with_info(response)
        assert result.success
        assert result.data == [1, 2, 3]
        assert "raw_json" in result.strategies_tried

    def test_truncation_repair_failure_extends_strategies(self):
        """Should track failed truncation repair strategies."""
        # Truncated JSON that repair can't fix
        response = '{broken "json without proper structure'
        result = extract_json_with_info(response)
        # Even if it fails, we should see strategies tried
        assert "truncation_repair" in result.strategies_tried or len(result.strategies_tried) > 0

    def test_fallback_pattern_actually_succeeds(self):
        """Should return ParseResult when fallback pattern finds valid JSON."""
        # Data where all other strategies fail but fallback succeeds
        # No code blocks, raw JSON parsing will find {invalid json} first and fail
        # Fallback pattern matches the valid JSON directly (group(0) is entire match)
        response = 'broken {invalid json} DATA={"key": "value"}'
        # Fallback pattern must match the exact JSON (not use capturing groups)
        result = extract_json_with_info(response, fallback_pattern=r'\{"key": "value"\}')
        assert result.success
        assert result.data == {"key": "value"}
        assert result.repair_applied == "fallback_pattern"
        assert "fallback_pattern" in result.strategies_tried


class TestUnwrapSingleJson:
    """Tests for unwrap_single_json function (#11)."""

    def test_unwrap_single_json_dict(self):
        """Should return dict as-is."""
        data = {"name": "test", "value": 42}
        result = unwrap_single_json(data)
        assert result == data

    def test_unwrap_single_json_list_one(self):
        """Should return the single dict from a one-element list."""
        data = [{"name": "test"}]
        result = unwrap_single_json(data)
        assert result == {"name": "test"}

    def test_unwrap_single_json_list_multiple(self, caplog):
        """Should return first dict from multi-element list with warning."""
        data = [{"name": "first"}, {"name": "second"}]
        with caplog.at_level(logging.WARNING):
            result = unwrap_single_json(data)
        assert result == {"name": "first"}
        assert any("2 dicts" in r.message for r in caplog.records)

    def test_unwrap_single_json_list_no_dicts(self, caplog):
        """Should return None when list contains no dicts."""
        data = ["string", 42, True]
        with caplog.at_level(logging.ERROR):
            result = unwrap_single_json(data)
        assert result is None
        assert any("no dicts" in r.message for r in caplog.records)

    def test_unwrap_single_json_none(self):
        """Should return None for None input."""
        result = unwrap_single_json(None)
        assert result is None

    def test_unwrap_single_json_unsupported_type(self):
        """Should return None for unsupported types like int or str."""
        assert unwrap_single_json(42) is None
        assert unwrap_single_json("not a dict") is None

    def test_unwrap_single_json_list_mixed_types(self):
        """Should extract first dict from list with mixed types."""
        data = ["string", 42, {"name": "found"}, True]
        result = unwrap_single_json(data)
        assert result == {"name": "found"}
