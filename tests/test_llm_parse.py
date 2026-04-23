import pytest

from mac.llm import _parse_json_loose


def test_plain_json():
    assert _parse_json_loose('{"a": 1}') == {"a": 1}


def test_fenced_json():
    text = """```json
{"a": 1, "b": [2, 3]}
```"""
    assert _parse_json_loose(text) == {"a": 1, "b": [2, 3]}


def test_embedded_json():
    text = 'Sure, here you go: {"x": "y"} hope that helps'
    assert _parse_json_loose(text) == {"x": "y"}


def test_array_embedded():
    text = "result:\n[1, 2, 3]\nthanks"
    assert _parse_json_loose(text) == [1, 2, 3]


def test_invalid_raises():
    with pytest.raises(ValueError):
        _parse_json_loose("not json at all")
