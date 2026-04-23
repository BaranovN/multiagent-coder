import pytest

from mac.agents import _coerce_programmer_output, _infer_run_command


def test_full_object_passthrough():
    data = {
        "files": [{"path": "main.py", "content": "print(1)"}],
        "build_command": None,
        "run_command": "python main.py",
    }
    out = _coerce_programmer_output(data, "Python")
    assert out["files"][0]["path"] == "main.py"
    assert out["run_command"] == "python main.py"


def test_bare_list_is_wrapped():
    data = [
        {"path": "fizz.py", "content": "print('hi')"},
        {"path": "util.py", "content": "x = 1"},
    ]
    out = _coerce_programmer_output(data, "Python")
    assert len(out["files"]) == 2
    assert out["run_command"] == "python fizz.py"
    assert out["build_command"] is None


def test_single_file_object_is_wrapped():
    data = {"path": "main.go", "content": "package main"}
    out = _coerce_programmer_output(data, "Go")
    assert out["files"][0]["path"] == "main.go"
    assert out["run_command"] == "go run ."


def test_missing_run_command_is_inferred():
    data = {"files": [{"path": "solve.py", "content": "print(0)"}]}
    out = _coerce_programmer_output(data, "Python")
    assert out["run_command"] == "python solve.py"


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        _coerce_programmer_output({"foo": "bar"}, "Python")
    with pytest.raises(ValueError):
        _coerce_programmer_output("not json", None)


def test_infer_languages():
    assert _infer_run_command([{"path": "a.rs"}], None) == "cargo run --release -q"
    assert _infer_run_command([{"path": "a.js"}], None) == "node a.js"
    assert _infer_run_command([{"path": "a.go"}], None) == "go run ."
    assert _infer_run_command([{"path": "a.py"}], None) == "python a.py"
