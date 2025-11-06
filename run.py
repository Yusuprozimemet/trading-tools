"""
Streamlit multi-page runner

Run this with:

    streamlit run run.py

This script discovers other .py files in the repository (excluding __init__.py and this runner)
and lets you execute any of them inside the same Streamlit session. It uses runpy to run the
selected file as if it were executed directly (so constructs under `if __name__ == '__main__'` run).

Notes / limitations:
- Some scripts may assume a fresh Streamlit session or particular working directory. This runner
  temporarily changes cwd to the script's directory when executing it.
- If a script mutates global state or sets Streamlit configuration at import-time, results may vary.
"""
from __future__ import annotations

import glob
import os
import runpy
import sys
import traceback
import subprocess
import socket
import time
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

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


def find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_streamlit_subprocess(path: str) -> Dict[str, Any]:
    """Start a separate Streamlit process serving the given script on a free port.

    Returns a dict with keys: process (Popen), port (int), log (str path), started_at (float)
    """
    port = find_free_port()
    # create a logfile path
    fd, log_path = tempfile.mkstemp(prefix="st_run_", suffix=".log")
    os.close(fd)
    log_file = open(log_path, "a", encoding="utf-8", errors="replace")

    cmd = [sys.executable, "-m", "streamlit", "run", path,
           "--server.port", str(port), "--server.runOnSave", "false"]
    # On Windows, CREATE_NEW_PROCESS_GROUP helps with termination
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            cwd=os.path.dirname(path) or None, creationflags=creationflags)
    return {"process": proc, "port": port, "log": log_path, "started_at": time.time(), "log_file": log_file}


def stop_streamlit_subprocess(entry: Dict[str, Any]) -> None:
    proc: subprocess.Popen = entry.get("process")
    log_file = entry.get("log_file")
    try:
        if proc and proc.poll() is None:
            proc.terminate()
            # wait briefly
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
    finally:
        try:
            if log_file and not log_file.closed:
                log_file.close()
        except Exception:
            pass


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
        Very easy: pick the page you want from the sidebar and click **Run selected** — each page
        will be served in its own Streamlit process and opens in a separate browser tab/window.

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
        "Scripts are served separately. Click 'Run selected' to start a dedicated Streamlit server for the selected page.")

    # map selected display back to file path
    selected_path = display_to_path.get(selected_display)

    # validate selected path exists and is a file
    if not os.path.exists(selected_path) or not os.path.isfile(selected_path):
        st.sidebar.error(
            "Selected path does not exist or is not a file. Please pick another script.")
        st.write(f"Selected path invalid: {selected_path}")
        return

    # Do not show the script filesystem path in the sidebar (user requested)

    run_now = st.sidebar.button("Run selected")
    stop_now = st.sidebar.button("Stop selected")

    # initialize session_state for subprocess management
    if "subprocesses" not in st.session_state:
        st.session_state.subprocesses = {}

    def _is_proc_running(entry: Dict[str, Any]) -> bool:
        p = entry.get("process")
        return p and p.poll() is None

    # Subprocess-only behavior: show subprocess status
    entry = st.session_state.subprocesses.get(selected_path)
    running = entry is not None and _is_proc_running(entry)
    if running:
        st.sidebar.success(f"Serving at http://localhost:{entry['port']}")
        st.sidebar.text(f"PID: {entry['process'].pid}")
    else:
        st.sidebar.info("Not running (subprocess)")

    # Start subprocess only when the user clicks the button
    if run_now and not running:
        st.write(f"### Starting (subprocess): {selected_display}")
        try:
            with st.spinner(f"Starting streamlit for {selected_display}..."):
                proc_entry = start_streamlit_subprocess(selected_path)
                st.session_state.subprocesses[selected_path] = proc_entry
        except Exception:
            st.error("Error while starting subprocess — see traceback below")
            st.code(traceback.format_exc())

    if stop_now and entry:
        st.write(f"### Stopping: {selected_display}")
        try:
            stop_streamlit_subprocess(entry)
        except Exception:
            st.error("Error while stopping process — see traceback below")
            st.code(traceback.format_exc())
        finally:
            st.session_state.subprocesses.pop(selected_path, None)

    # show logs if available
    entry = st.session_state.subprocesses.get(selected_path)
    if entry and entry.get("log"):
        st.write("---")
        with st.expander("Show server log"):
            try:
                log_path = entry.get("log")
                # show last 50000 chars
                text = Path(log_path).read_text(
                    encoding="utf-8", errors="replace")
                st.text_area("Log", value=text[-50000:], height=300)
            except Exception as e:
                st.write(f"Could not read log: {e}")

    # Removed extra troubleshooting info box per user request.


if __name__ == "__main__":
    main()
