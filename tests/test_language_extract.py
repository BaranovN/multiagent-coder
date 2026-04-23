from mac.agents import _extract_language


def test_heading_followed_by_prose():
    spec = """# Problem
...

### Recommended target language
Python is recommended for this task.
"""
    assert _extract_language(spec) == "Python"


def test_inline_colon():
    spec = "Target language: Rust (because bounds are tight)."
    assert _extract_language(spec) == "Rust"


def test_bold_markdown():
    spec = "**Target language:** *TypeScript*"
    assert _extract_language(spec) == "TypeScript"


def test_global_fallback():
    spec = "We should solve this in Go to keep the binary small."
    assert _extract_language(spec) == "Go"


def test_plain_c_with_word_boundary():
    spec = "Recommended target language: C\nUse the standard library."
    assert _extract_language(spec) == "C"


def test_no_language_returns_none():
    spec = "Some totally unrelated text"
    assert _extract_language(spec) is None


def test_cpp_is_matched():
    spec = "Recommended target language: C++"
    assert _extract_language(spec) == "C++"
