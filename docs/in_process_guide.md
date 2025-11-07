# Making pages safe to run in-process

When the multi-runner executes a page inside the same Streamlit process (in-process), the page's
module is imported or executed inside an existing Streamlit session. This means top-level Streamlit
calls (like `st.sidebar.*`, `st.title`, etc.) run immediately at import time and can conflict with
other pages or share session state in unexpected ways.

Recommended pattern

- Wrap UI code into a `main()` function and guard execution with:

```python
if __name__ == "__main__":
    main()
```

- Keep module-level constants, imports and helper functions at top-level, but avoid calling Streamlit
  API functions at import time.

Example (see `examples/inprocess_template.py`):

```python
import streamlit as st


def main():
    st.title("My page")
    st.sidebar.selectbox("Option", ["A", "B"])  # OK inside main


if __name__ == "__main__":
    main()
```

Quick scan

Use the provided scanner to find candidate files that may need refactoring:

```powershell
python tools/scan_inprocess.py
```

This is a heuristic. Files flagged by the scanner should be inspected manually and converted to the
pattern above where appropriate.

Notes

- Some scripts intentionally rely on a fresh Streamlit process (for example they set global state,
  or expect that `st.session_state` is empty). Those should either be converted to the `main()` pattern
  or be kept as separate services in production.
- The runner keeps the existing "Subprocess" mode for local development where starting a separate
  Streamlit server and opening a new browser tab works.
