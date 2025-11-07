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
        if bname.lower() == "terminal_trading.py":
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


def categorize_scripts(scripts: List[Tuple[str, str]], mapping: Dict[str, str]) -> Dict[str, List[Tuple[str, str, str]]]:
    """Organize scripts into categories based on their folder structure.
    
    Returns: Dict[category_name, List[(display_name, rel_path, abs_path)]]
    """
    categories: Dict[str, List[Tuple[str, str, str]]] = {
        "📊 Analysis & Charts": [],
        "💼 Portfolio & Finance": [],
        "🎯 Trading Strategies": [],
        "⚡ Trading Execution": [],
        "🔔 Alerts & Monitoring": [],
    }
    
    for rel, abspath in scripts:
        rel_norm = Path(rel).as_posix()
        display = mapping.get(rel_norm) or mapping.get(rel) or Path(rel).stem
        
        # Categorize based on folder structure
        if "chart" in rel.lower():
            categories["📊 Analysis & Charts"].append((display, rel, abspath))
        elif "portofolio" in rel.lower() or "finance" in rel.lower():
            categories["💼 Portfolio & Finance"].append((display, rel, abspath))
        elif "strategies" in rel.lower():
            categories["🎯 Trading Strategies"].append((display, rel, abspath))
        elif "trading" in rel.lower():
            categories["⚡ Trading Execution"].append((display, rel, abspath))
        elif "alert" in rel.lower():
            categories["🔔 Alerts & Monitoring"].append((display, rel, abspath))
        else:
            # Default category for uncategorized items
            if "📚 Other Tools" not in categories:
                categories["📚 Other Tools"] = []
            categories["📚 Other Tools"].append((display, rel, abspath))
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def main() -> None:
    st.set_page_config(
        page_title="Trading Tools Suite",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = find_scripts(base_dir)

    if not scripts:
        st.sidebar.info("No candidate .py scripts found in the repository.")
        st.write("No scripts detected.")
        return

    # prepare display names (allow mapping via streamlit_pages.json)
    script_names = [s[0] for s in scripts]
    mapping = load_display_mapping(base_dir)
    
    # Categorize scripts
    categories = categorize_scripts(scripts, mapping)
    
    # ========== HERO SECTION ==========
    st.markdown("""
        <style>
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #1e88e5 0%, #00acc1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .hero-subtitle {
            font-size: 1.3rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .feature-card h3 {
            color: white;
            margin-bottom: 0.5rem;
        }
        .stat-box {
            background: #f8f9fa;
            border-left: 4px solid #1e88e5;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        .category-header {
            background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
            padding: 0.8rem;
            border-radius: 8px;
            border-left: 4px solid #1e88e5;
            margin: 1rem 0 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Hero title and subtitle
    st.markdown('<h1 class="hero-subtitle">Professional Trading Analysis & Strategy Backtesting Platform</h1>', unsafe_allow_html=True)
    
    # ========== FEATURE HIGHLIGHTS ==========
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="🎯 Strategies", value=len([s for cat in categories.values() for s in cat if "Strategies" in cat or any("strategies" in path.lower() for _, path, _ in cat)]))
    with col2:
        st.metric(label="📊 Analysis Tools", value=len(categories.get("📊 Analysis & Charts", [])))
    with col3:
        st.metric(label="⚡ Live Trading", value=len(categories.get("⚡ Trading Execution", [])))
    with col4:
        st.metric(label="🛠️ Total Tools", value=len(scripts))
    
    st.markdown("---")
    
    # ========== WELCOME MESSAGE ==========
    with st.expander("ℹ️ **Quick Start Guide**", expanded=False):
        st.markdown("""
        ### Welcome to Trading Tools Suite! 👋
        
        This comprehensive platform provides professional-grade trading analysis and backtesting tools.
        
        **How to Use:**
        1. 📋 Browse available tools in the **sidebar** organized by category
        2. ✅ Select any tool from the dropdown menu
        3. ▶️ Click **"Run Selected Tool"** to launch it
        4. 📊 Interact with the tool and analyze results in real-time
        5. 🔄 Use **"Clear Output"** to reset and try another tool
        
        **Features:**
        - ✨ **20+ Trading Strategies** - Bollinger Bands, RSI, Momentum, Pairs Trading & More
        - 📈 **Advanced Analytics** - Monte Carlo Simulations, Correlation Analysis, Backtesting
        - 💼 **Portfolio Management** - Optimization, Risk Analysis, Performance Tracking
        - ⚡ **Live Trading Integration** - Alpaca API, Real-time Data, Automated Execution
        - 🔔 **Smart Alerts** - Custom notifications for trading signals
        
        **For Contributors:**
        - Add new tools by creating `.py` files in appropriate folders
        - Update `streamlit_pages.json` for custom display names
        - See `README.md` for detailed contribution guidelines
        """)
    
    # ========== SIDEBAR CONFIGURATION ==========
    # Display logo at the top of the sidebar (if present)
    logo_path = os.path.join(base_dir, "image", "logo.jpeg")
    if os.path.exists(logo_path):
        try:
            st.sidebar.image(logo_path, width=140)
        except Exception:
            # If image fails to load for any reason, silently continue
            pass

    st.sidebar.markdown("### 🎯 Select a Tool")
    st.sidebar.markdown("Browse by category and choose a tool to run:")
    st.sidebar.markdown("---")
    
    # Build categorized display list with visual separators
    display_to_path: Dict[str, str] = {}
    display_list: List[str] = []
    
    for category_name, items in categories.items():
        # Add category as a disabled option (visual separator)
        category_label = f"━━━ {category_name} ━━━"
        display_list.append(category_label)
        display_to_path[category_label] = ""  # Empty path for category headers
        
        # Add items in this category
        for display, rel, abspath in sorted(items, key=lambda x: x[0]):
            # Add indentation for visual hierarchy
            display_item = f"  ▸ {display}"
            display_to_path[display_item] = abspath
            display_list.append(display_item)
    
    # Category selection in sidebar
    selected_display = st.sidebar.selectbox(
        "Choose a tool:",
        display_list,
        format_func=lambda x: x,
        help="Select any tool from the categorized list below"
    )
    
    # map selected display back to file path
    selected_path = display_to_path.get(selected_display, "")
    
    # Show tool info in sidebar if valid selection
    if selected_path and os.path.exists(selected_path):
        tool_name = selected_display.replace("  ▸ ", "").strip()
        st.sidebar.success(f"✓ Selected: **{tool_name}**")
    elif not selected_path:
        st.sidebar.info("👆 Please select a tool from the list above")
    
    st.sidebar.markdown("---")
    
    # Action buttons
    col_run, col_clear = st.sidebar.columns(2)
    with col_run:
        run_now = st.button("▶️ Run Selected", key="run_button", use_container_width=True, type="primary")
    with col_clear:
        clear_now = st.button("🔄 Clear Output", key="clear_button", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 **Tip:** Use the expander above for Quick Start Guide")
    st.sidebar.caption("📖 Check README.md for detailed documentation")

    # validate selected path exists and is a file
    if selected_path and (not os.path.exists(selected_path) or not os.path.isfile(selected_path)):
        st.sidebar.error("Selected path does not exist or is not a file. Please pick another script.")
        st.error(f"⚠️ Selected path invalid: {selected_path}")
        return

    # initialize session_state for inline script execution
    if "current_script" not in st.session_state:
        st.session_state.current_script = None
    if "script_output" not in st.session_state:
        st.session_state.script_output = None
    if "script_name" not in st.session_state:
        st.session_state.script_name = None

    # Handle clearing output
    if clear_now:
        st.session_state.current_script = None
        st.session_state.script_output = None
        st.session_state.script_name = None
        st.rerun()

    # Run script inline when button is clicked
    if run_now and selected_path:
        st.session_state.current_script = selected_path
        st.session_state.script_name = selected_display.replace("  ▸ ", "").strip()
        st.rerun()

    # Execute the selected script if set
    if st.session_state.current_script:
        # Display running tool header
        st.markdown(f"""
            <div style="background: linear-gradient(90deg, #1e88e5 0%, #00acc1 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
                <h2 style="color: white; margin: 0;">
                    ▶️ {st.session_state.script_name}
                </h2>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Tool is running... Interact with the controls below
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        try:
            run_script(st.session_state.current_script)
        except Exception as e:
            st.error(f"❌ **Error running script:** {str(e)}")
            with st.expander("🔍 View Error Details"):
                st.code(traceback.format_exc())
    else:
        # Show tool gallery when no tool is running
        st.markdown("## 🛠️ Available Tools")
        st.markdown("Select a tool from the sidebar to get started, or browse the categories below:")
        
        # Display tools by category in main area
        for category_name, items in categories.items():
            st.subheader(category_name)
            cols = st.columns(3)
            
            for idx, (display, rel, _) in enumerate(sorted(items, key=lambda x: x[0])):
                folder = os.path.dirname(rel) or "root"
                with cols[idx % 3]:
                    st.info(f"**{display}**\n\n📁 {folder}")


if __name__ == "__main__":
    main()
