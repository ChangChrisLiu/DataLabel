"""``python -m tda ...`` -- the same command line as ``python -m tda.cli ...``.

Both spellings work and always did in spirit; only the second one existed, and
it is the one that runs ``cli.py`` as ``__main__``. This module is the ordinary
way in: ``tda.cli`` stays a plain import, imported exactly once.
"""
from __future__ import annotations

from tda.cli import main

if __name__ == "__main__":  # pragma: no cover - exercised by tests/test_cli_subprocess.py
    raise SystemExit(main())
