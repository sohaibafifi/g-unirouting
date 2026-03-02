from __future__ import annotations

import subprocess
import sys

from pathlib import Path


def main() -> None:
    shared_script = Path(__file__).resolve().parents[1] / "text_explanations.py"
    cmd = [sys.executable, str(shared_script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
