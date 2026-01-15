"""Tests for the JSON parsing utility."""

from pydantic import BaseModel

from utils.json_parser import extract_json, extract_json_list, parse_json_to_model


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
