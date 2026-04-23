"""A minimal subprocess-based sandbox for running untrusted code.

This is not a security boundary — it's a correctness boundary. For real
isolation swap the backend for Docker (`docker run --rm --network=none ...`)
or Firecracker. See the roadmap in README.

Responsibilities:
  * Materialize an `{path: content}` file map into a fresh tmpdir.
  * Run a build command (optional) and a run command with a stdin payload.
  * Enforce wall-clock timeouts; capture stdout/stderr.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


async def _run(cmd: str, *, cwd: Path, stdin: str = "", timeout: int) -> ExecResult:
    import time

    t0 = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        return ExecResult(127, "", f"shell not found: {e}", 0)

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(stdin.encode("utf-8")), timeout=timeout
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        dur = int((time.monotonic() - t0) * 1000)
        return ExecResult(124, "", f"timeout after {timeout}s running: {cmd}", dur, True)

    dur = int((time.monotonic() - t0) * 1000)
    return ExecResult(
        exit_code=proc.returncode or 0,
        stdout=stdout_b.decode("utf-8", "replace"),
        stderr=stderr_b.decode("utf-8", "replace"),
        duration_ms=dur,
    )


def _write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        # Prevent path traversal out of the sandbox root.
        safe_rel = Path(rel)
        if safe_rel.is_absolute() or ".." in safe_rel.parts:
            raise ValueError(f"unsafe path in file map: {rel}")
        target = root / safe_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


@dataclass
class Sandbox:
    timeout: int = 30
    keep_workdir: bool = False

    async def run(
        self,
        files: dict[str, str],
        *,
        build_command: str | None,
        run_command: str,
        stdin: str = "",
    ) -> tuple[ExecResult | None, ExecResult]:
        """Materialize files, optionally build, then run once. Returns (build, run)."""
        workdir = Path(tempfile.mkdtemp(prefix="mac-sandbox-"))
        logger.debug("sandbox workdir: %s", workdir)
        try:
            _write_files(workdir, files)
            build_res: ExecResult | None = None
            if build_command:
                build_res = await _run(build_command, cwd=workdir, timeout=self.timeout)
                if build_res.exit_code != 0:
                    return build_res, ExecResult(0, "", "build failed; run skipped", 0)
            run_res = await _run(run_command, cwd=workdir, stdin=stdin, timeout=self.timeout)
            return build_res, run_res
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    async def run_many(
        self,
        files: dict[str, str],
        *,
        build_command: str | None,
        run_command: str,
        stdins: list[str],
    ) -> tuple[ExecResult | None, list[ExecResult]]:
        """Build once, then run the same program against many stdin payloads."""
        workdir = Path(tempfile.mkdtemp(prefix="mac-sandbox-"))
        try:
            _write_files(workdir, files)
            build_res: ExecResult | None = None
            if build_command:
                build_res = await _run(build_command, cwd=workdir, timeout=self.timeout)
                if build_res.exit_code != 0:
                    return build_res, []
            runs: list[ExecResult] = []
            for stdin in stdins:
                r = await _run(run_command, cwd=workdir, stdin=stdin, timeout=self.timeout)
                runs.append(r)
            return build_res, runs
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)


def quote_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)
