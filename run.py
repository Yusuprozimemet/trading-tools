"""
Streamlit multi-page runner

Run this with:

    streamlit run run.py

This script discovers other .py files in the repository (excluding __init__.py and this runner)
and lets you execute any of them inside the same Streamlit session. It uses runpy to run the
selected file as if it were executed directly (so constructs under `if __name__ == '__main__'` run).

All selected scripts run directly within this Streamlit app - no separate processes needed.
"""
from __future__ import annotations

import glob
import os
import runpy
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st


def find_scripts(base_dir: str) -> List[Tuple[str, str]]:
    """Return list of (relpath, abspath) for candidate python scripts."""
    pattern = os.path.join(base_dir, "**", "*.py")
    files = glob.glob(pattern, recursive=True)
    out: List[Tuple[str, str]] = []
    this_file = os.path.abspath(__file__)
    for f in sorted(files):
        af = os.path.abspath(f)
        if af == this_file:
            continue
        if "__pycache__" in af:
            continue
        # exclude package init files and specific scripts we don't want listed
        bname = os.path.basename(af)
        if bname == "__init__.py":
            continue
        if bname.lower() == "alert.py":
            continue
        rel = os.path.relpath(af, base_dir)
        out.append((rel, af))
    return out


def load_display_mapping(base_dir: str) -> Dict[str, str]:
    """Load optional mapping file `streamlit_pages.json` at repo root.

    Mapping format: {"relative/path/to/script.py": "Display Name", ...}
    Returns empty dict if file doesn't exist or can't be parsed.
    """
    mapping_file = os.path.join(base_dir, "streamlit_pages.json")
    if not os.path.exists(mapping_file):
        return {}
    try:
        import json

        with open(mapping_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # normalize keys to POSIX-style relative paths so lookup works on Windows and Unix
            out: Dict[str, str] = {}
            for k, v in data.items():
                try:
                    key_norm = Path(str(k)).as_posix()
                except Exception:
                    key_norm = str(k)
                out[key_norm] = str(v)
            return out
    except Exception:
        pass
    return {}


def run_script(path: str) -> None:
    """Execute the script located at path as __main__ and capture exceptions to Streamlit."""
    old_cwd = os.getcwd()
    script_dir = os.path.dirname(path) or old_cwd
    try:
        os.chdir(script_dir)
        # run as main so code under if __name__ == '__main__' runs
        runpy.run_path(path, run_name="__main__")
    finally:
        os.chdir(old_cwd)


def main() -> None:
    st.set_page_config(page_title="Streamlit Multi-runner", layout="wide")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = find_scripts(base_dir)

    st.sidebar.title("Pages")
    if not scripts:
        st.sidebar.info("No candidate .py scripts found in the repository.")
        st.write("No scripts detected.")
        return

    # Short manual: concise description and contribution pointers
    st.title("Streamlit Multi-runner")
    st.markdown(
        """
        Select a page from the sidebar and click **Run selected** to run it directly within this app.
        All output, charts, and interactions will appear below.

        This app is designed so contributors can add new pages (scripts) under the project folders.
        See the repository `README.md` for contribution guidelines and how to add display names
        (optional `streamlit_pages.json` mapping).
        """
    )

    # prepare display names (allow mapping via streamlit_pages.json)
    script_names = [s[0] for s in scripts]
    mapping = load_display_mapping(base_dir)

    # Build display name list and map back to absolute path
    display_to_path: Dict[str, str] = {}
    display_list: List[str] = []
    seen: Dict[str, int] = {}
    for rel, abspath in scripts:
        # normalize rel to POSIX style for consistent mapping lookup
        rel_norm = Path(rel).as_posix()
        display = mapping.get(rel_norm) or mapping.get(rel) or Path(rel).stem
        # disambiguate duplicates
        if display in seen:
            seen[display] += 1
            display = f"{display} ({seen[display]})"
        else:
            seen[display] = 1
        display_to_path[display] = abspath
        display_list.append(display)

    selected_display = st.sidebar.selectbox(
        "Select script to run", display_list)
    st.sidebar.write("---")
    st.sidebar.caption(
        "Click 'Run selected' to execute the selected page directly in this app.")

    # map selected display back to file path
    selected_path = display_to_path.get(selected_display)

    # validate selected path exists and is a file
    if not os.path.exists(selected_path) or not os.path.isfile(selected_path):
        st.sidebar.error(
            "Selected path does not exist or is not a file. Please pick another script.")
        st.write(f"Selected path invalid: {selected_path}")
        return

    # initialize session_state for inline script execution
    if "current_script" not in st.session_state:
        st.session_state.current_script = None
    if "script_output" not in st.session_state:
        st.session_state.script_output = None

    run_now = st.sidebar.button("Run selected", key="run_button")
    clear_now = st.sidebar.button("Clear output", key="clear_button")

    # Handle clearing output
    if clear_now:
        st.session_state.current_script = None
        st.session_state.script_output = None
        st.rerun()

    # Run script inline when button is clicked
    if run_now:
        st.session_state.current_script = selected_path
        st.sidebar.success(f"✓ Running: {selected_display}")

    # Execute the selected script if set
    if st.session_state.current_script:
        st.write(f"### Running: {selected_display}")
        st.divider()
        try:
            run_script(st.session_state.current_script)
        except Exception as e:
            st.error(f"Error running script: {str(e)}")
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
