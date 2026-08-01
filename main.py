# AI Assistance Attribution:
# This file was generated with assistance from OpenAI ChatGPT:
# https://chatgpt.com/share/6a6d47a0-1a54-83e8-91a4-2f8c8b673fe5

"""Command-line entry point for launching the Streamlit application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final


STREAMLIT_APP_PATH: Final[Path] = Path("app/app.py")


def build_streamlit_command(app_path: Path) -> list[str]:
    """Build the Streamlit launch command using the active Python interpreter."""

    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]


def launch_streamlit_app(app_path: Path) -> int:
    """Launch the Streamlit application and return its process exit code."""

    if not app_path.is_file():
        print(
            f"Streamlit application file not found: {app_path}",
            file=sys.stderr,
        )
        return 1

    command = build_streamlit_command(app_path)

    try:
        completed_process = subprocess.run(
            command,
            check=False,
        )
    except OSError as error:
        print(
            f"Failed to launch Streamlit: {error}",
            file=sys.stderr,
        )
        return 1

    return completed_process.returncode


def main() -> int:
    """Launch the Context-to-Learning Streamlit application."""

    return launch_streamlit_app(STREAMLIT_APP_PATH)


if __name__ == "__main__":
    raise SystemExit(main())