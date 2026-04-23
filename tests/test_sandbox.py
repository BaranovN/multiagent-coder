import pytest

from mac.sandbox import Sandbox


@pytest.mark.asyncio
async def test_python_hello_runs():
    sb = Sandbox(timeout=10)
    build, run = await sb.run(
        {"main.py": "print('hello', input().strip())\n"},
        build_command=None,
        run_command="python3 main.py",
        stdin="world\n",
    )
    assert build is None
    assert run.exit_code == 0
    assert "hello world" in run.stdout


@pytest.mark.asyncio
async def test_timeout_is_enforced():
    sb = Sandbox(timeout=2)
    _, run = await sb.run(
        {"main.py": "import time; time.sleep(10)\n"},
        build_command=None,
        run_command="python3 main.py",
    )
    assert run.timed_out
    assert run.exit_code != 0


@pytest.mark.asyncio
async def test_path_traversal_rejected():
    sb = Sandbox(timeout=2)
    with pytest.raises(ValueError):
        await sb.run(
            {"../escape.py": "print(1)\n"},
            build_command=None,
            run_command="python3 main.py",
        )
