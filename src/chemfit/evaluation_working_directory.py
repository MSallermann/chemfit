from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

logger = logging.getLogger(__name__)


def _is_safe_relative_path(path: Path) -> bool:
    """Return whether a relative path stays below its eventual parent."""
    if path.is_absolute():
        return False

    # Validate lexically: the unique evaluation directory does not exist yet,
    # and resolving the path would make the result depend on the current cwd.
    depth = 0
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                return False
        else:
            depth += 1
    return True


class EvaluationWorkingDirectory:
    """Configuration and lifecycle policy for isolated evaluation directories."""

    def __init__(
        self,
        base_working_directory: Path | str,
        *,
        delete_temp_workdirs: bool = True,
        keep_temp_workdir_after_crash: bool = True,
    ) -> None:
        """
        Configure per-evaluation working directories.

        Args:
            base_working_directory: Parent below which unique directories are
                created.
            delete_temp_workdirs: Whether directories are removed after an
                evaluation. ``keep_temp_workdir_after_crash`` takes precedence
                when an evaluation fails.
            keep_temp_workdir_after_crash: Whether failed evaluation
                directories are retained for inspection.

        """
        self.base_working_directory = Path(base_working_directory)
        self.delete_temp_workdirs = delete_temp_workdirs
        self.keep_temp_workdir_after_crash = keep_temp_workdir_after_crash

    @staticmethod
    def relative_path(
        path: Path | str,
        *,
        description: str = "Output path",
    ) -> Path:
        """Normalize one path and ensure it stays in an evaluation directory."""
        normalized = Path(path)
        if normalized.is_absolute():
            msg = f"{description} must be relative to the evaluation working directory."
            raise ValueError(msg)
        if not _is_safe_relative_path(normalized):
            msg = (
                "Relative path is not safe because it leaves the current directory "
                "(too many `..`)."
            )
            raise ValueError(msg)
        return normalized

    @staticmethod
    def relative_paths(
        paths: Iterable[Path | str],
        *,
        description: str = "Output paths",
    ) -> tuple[Path, ...]:
        """Normalize paths and ensure they cannot escape an evaluation directory."""
        return tuple(
            EvaluationWorkingDirectory.relative_path(
                path,
                description=description,
            )
            for path in paths
        )

    def create(self) -> Path:
        """Create and return a uniquely named evaluation directory."""
        workdir = self.base_working_directory / str(uuid.uuid4())
        workdir.mkdir(exist_ok=False, parents=True)
        return workdir

    def cleanup(self, workdir: Path, *, failed: bool) -> str:
        """Apply the cleanup policy and return an exception-note suffix."""
        keep = not self.delete_temp_workdirs or (
            failed and self.keep_temp_workdir_after_crash
        )
        if keep:
            if failed:
                return f"\nKeeping temporary workdir '{workdir}' for inspection"
            return ""

        try:
            shutil.rmtree(workdir)
        except Exception as cleanup_exception:
            if not failed:
                raise
            cleanup_msg = (
                f"Could not remove temporary workdir '{workdir}' after the "
                f"evaluation failed: {cleanup_exception}."
            )
            logger.warning(cleanup_msg)
            return f"\n{cleanup_msg}"
        return ""

    @contextmanager
    def isolated_workdir(self) -> Generator[Path, None, None]:
        """Create an evaluation directory and apply cleanup when it exits."""
        workdir = self.create()
        try:
            yield workdir
        except BaseException as exception:
            # Preserve the evaluation failure if best-effort cleanup also fails.
            note = self.cleanup(workdir, failed=True)
            if note and hasattr(exception, "add_note"):
                exception.add_note(note)
            raise
        else:
            self.cleanup(workdir, failed=False)
