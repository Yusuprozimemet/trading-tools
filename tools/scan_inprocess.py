"""
Scan repository python files for potential in-process incompatibilities.
Heuristic: flag files that import `streamlit` (or use `st.`) but do not define a `main()` function or an
`if __name__ == '__main__'` guard. Those files may run UI code at import time and can conflict when
executed inside an existing Streamlit session.

Usage:
    python tools/scan_inprocess.py

This prints a list of candidate files to review/convert.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

PY_PATTERN = re.compile(r".*\.py$")


def looks_like_streamlit_user(code: str) -> bool:
    return "import streamlit" in code or "from streamlit" in code or "st." in code


def has_main_guard(code: str) -> bool:
    return "if __name__" in code or "def main(" in code


def main():
    print(f"Scanning {ROOT} for in-process issues...\n")
    candidates = []
    for p in ROOT.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        if "__pycache__" in str(p):
            continue
        if p.name == "run.py":
            continue
        try:
            code = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if looks_like_streamlit_user(code) and not has_main_guard(code):
            candidates.append(p.relative_to(ROOT))

    if not candidates:
        print("No obvious in-process issues found.\n")
        return

    print("Files that may need refactoring to run in-process (review and wrap UI in a main() or add guard):\n")
    for c in sorted(candidates):
        print(f" - {c}")


if __name__ == "__main__":
    main()
